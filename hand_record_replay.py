"""
hand_record_replay.py — Record hand joint data with a pinch and replay it in Unity.

Optional add-on for main_hands_simple.py, enabled with:

    python main_hands_simple.py --record-replay

Workflow
--------
  1. Record a hand: pinch its thumb and middle fingertips together — the hand turns
     red (Open3D + headset) and every joint is recorded. Pinch again to stop; the
     recording is saved as a new JSON file in recordings/. Each hand records
     independently, so both can record at once. Requires the world to be locked.
  2. Replay: press L or R in the OpenCV window to play the newest saved recording
     of that hand once in Unity as a ghost skeleton, anchored to marker 100.

Unity side: HandReplayReceiver on WorldRoot (REPLAY_PORT). Settings live in
main_settings.py under "Hand recording / replay".
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import zmq
from scipy.spatial.transform import Rotation as ScipyR

import main_settings as cfg
from main_hands_simple import (
    T_to_unity, to_world, _HAND_BONES, _N_JOINTS, _JOINT_GROUP_ORDER,
)

_JOINT_NAMES = ["Wrist", "Palm"] + [f"{finger}{seg}"
                                    for finger in _JOINT_GROUP_ORDER[2:]
                                    for seg in ("1", "2", "3", "Tip")]
_THUMB_TIP, _MIDDLE_TIP = _JOINT_NAMES.index("ThumbTip"), _JOINT_NAMES.index("MiddleTip")


# =============================================================================
# Pose helpers
# =============================================================================

def extract_joint_poses(hand_block):
    """Flatten a Unity hand block into (22, 3) positions and (22, 4) xyzw rotations in
    Unity tracking space, or None. Joints Unity could not resolve are zero / identity."""
    if not hand_block or not hand_block.get("groups"):
        return None
    pos, rot = [], []
    for name in _JOINT_GROUP_ORDER:
        for pose in hand_block["groups"].get(name) or []:
            p = (pose or {}).get("position") or {}
            r = (pose or {}).get("rotation") or {}
            pos.append([p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)])
            rot.append([r.get("x", 0.0), r.get("y", 0.0), r.get("z", 0.0), r.get("w", 1.0)])
    if len(pos) != _N_JOINTS:
        return None
    return np.array(pos, dtype=np.float64), np.array(rot, dtype=np.float64)


def tracking_to_root_local(pos_unity: np.ndarray, rot_unity: np.ndarray,
                           T_world_tracking: np.ndarray):
    """Unity tracking-space joint poses → poses local to Unity's WorldRoot (Unity axes).

    WorldRoot is placed at T_to_unity(inv(T_wt)), so a world-frame point maps to
    WorldRoot-local Unity coordinates by the same Y/Z swap used everywhere else, and
    rotations are re-expressed relative to WorldRoot's rotation.
    """
    local_pos = to_world(pos_unity, T_world_tracking)[:, [0, 2, 1]]
    _, root_rot, _ = T_to_unity(np.linalg.inv(T_world_tracking))
    local_rot = (ScipyR.from_quat(root_rot).inv() * ScipyR.from_quat(rot_unity)).as_quat()
    return local_pos, local_rot


# =============================================================================
# Recording
# =============================================================================

class HandRecorder:
    """Records one hand. A thumb-middle pinch toggles recording; stopping saves a
    new JSON file. Frames are stored local to Unity's WorldRoot (marker 100), so a
    replay lands on the same physical spot even after a relock."""

    def __init__(self, side: str):
        self.side = side                  # "left" / "right"
        self.recording = False
        self._pinched = False             # pinch must open past PINCH_OFF_DIST before it can toggle again
        self._frames: list = []
        self._t0 = None
        self._last_ts = None

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def update(self, hand_block, T_world_tracking, unity_ts) -> None:
        """Feed the latest message for this hand (None if the hand is not tracked)."""
        poses = extract_joint_poses(hand_block)
        if poses is None:
            return   # keep the current state through tracking dropouts
        pos, rot = poses

        gap = float(np.linalg.norm(pos[_THUMB_TIP] - pos[_MIDDLE_TIP]))
        if not self._pinched and gap < cfg.PINCH_ON_DIST:
            self._pinched = True
            if self.recording:
                self.stop()
            else:
                self._start(T_world_tracking)
        elif self._pinched and gap > cfg.PINCH_OFF_DIST:
            self._pinched = False

        ts = unity_ts if unity_ts is not None else time.time()
        if self.recording and T_world_tracking is not None and ts != self._last_ts:
            if self._t0 is None:
                self._t0 = ts
            local_pos, local_rot = tracking_to_root_local(pos, rot, T_world_tracking)
            self._frames.append({
                "t": round(float(ts - self._t0), 4),
                "positions": np.round(local_pos, 5).tolist(),
                "rotations": np.round(local_rot, 5).tolist(),
            })
            self._last_ts = ts

    def _start(self, T_world_tracking) -> None:
        if T_world_tracking is None:
            print(f"[Record] {self.side} hand: lock the world to marker "
                  f"#{cfg.ANCHOR_MARKER_ID} before recording.")
            return
        self.recording = True
        self._frames, self._t0, self._last_ts = [], None, None
        print(f"[Record] {self.side} hand: recording...")

    def stop(self) -> "Path | None":
        """Stop recording and save; returns the new file (None if nothing was recorded)."""
        self.recording = False
        if not self._frames:
            print(f"[Record] {self.side} hand: stopped, no frames captured.")
            return None
        cfg.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = cfg.RECORDINGS_DIR / f"{self.side}_{stamp}.json"
        n = 1
        while path.exists():
            path = cfg.RECORDINGS_DIR / f"{self.side}_{stamp}_{n}.json"
            n += 1
        path.write_text(json.dumps({
            "hand": self.side,
            "created": datetime.now().isoformat(timespec="seconds"),
            "frame": (f"Local to Unity WorldRoot (ArUco marker #{cfg.ANCHOR_MARKER_ID}); "
                      "Unity axes (x right, y up, z forward), metres; rotations xyzw"),
            "joint_names": _JOINT_NAMES,
            "bones": _HAND_BONES.tolist(),
            "frame_count": len(self._frames),
            "duration_s": self._frames[-1]["t"],
            "frames": self._frames,
        }))
        print(f"[Record] {self.side} hand: saved {len(self._frames)} frames "
              f"({self._frames[-1]['t']:.1f} s) -> {path.name}")
        self._frames = []
        return path


def latest_recording(side: str) -> "Path | None":
    """Newest saved recording for 'left' / 'right', or None."""
    files = list(cfg.RECORDINGS_DIR.glob(f"{side}_*.json"))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


# =============================================================================
# Python → Unity
# =============================================================================

class ReplayLink:
    """Python → Unity HandReplayReceiver: replays plus each hand's live recording state."""

    def __init__(self, ip: str, port: int):
        self._pub = zmq.Context.instance().socket(zmq.PUB)
        self._pub.connect(f"tcp://{ip}:{port}")
        self._recording = {"left": False, "right": False}
        self._last_state_send = 0.0
        print(f"[Replay] PUB → tcp://{ip}:{port}")

    def set_recording(self, side: str, on: bool) -> None:
        if self._recording[side] != on:
            self._recording[side] = on
            self._send_state(side)

    def refresh(self) -> None:
        """Repeat both recording states so a Unity subscriber that connects late catches up."""
        if time.monotonic() - self._last_state_send >= cfg.RECORD_STATE_REFRESH_S:
            for side in self._recording:
                self._send_state(side)

    def _send_state(self, side: str) -> None:
        self._pub.send_string(json.dumps({"type": "record_state", "hand": side,
                                          "recording": self._recording[side],
                                          "color": cfg.RECORDING_COLOR}))
        self._last_state_send = time.monotonic()

    def replay(self, path: Path) -> None:
        data = json.loads(path.read_text())
        frames = [{"t": f["t"], "p": [c for p in f["positions"] for c in p]}
                  for f in data["frames"]]
        self._pub.send_string(json.dumps({"type": "replay", "hand": data["hand"],
                                          "frames": frames}))
        print(f"[Replay] {data['hand']} hand: {path.name} "
              f"({len(frames)} frames, {data.get('duration_s', 0):.1f} s)")

    def close(self):
        for side in self._recording:
            self._recording[side] = False
            self._send_state(side)
        self._pub.close(200)   # brief linger so the final "not recording" reaches Unity


# =============================================================================
# Add-on entry point used by main_hands_simple.MainScene
# =============================================================================

class HandRecordReplay:
    """Everything main_hands_simple needs: per-frame update, status text, L/R keys, close."""

    KEYS = "  L/R=replay"

    def __init__(self, quest_ip: str, vis):
        self._vis = vis
        self.link = ReplayLink(quest_ip, cfg.REPLAY_PORT)
        self.recorders = {"left": HandRecorder("left"), "right": HandRecorder("right")}

    def update(self, hands, left, right, T_wt) -> None:
        """hands: HandDataReceiver; left/right: filtered world joints (None = not tracked)."""
        for side, pts, key in (("left", left, "LeftHand"), ("right", right, "RightHand")):
            rec = self.recorders[side]
            was_recording = rec.recording
            # Only feed hands that passed the head-distance filter in world_joints().
            rec.update(hands.hand_block(key) if pts is not None else None,
                       T_wt, hands.unity_timestamp)
            if rec.recording != was_recording:
                self._show_recording(side, rec.recording)
        self.link.refresh()

    def _show_recording(self, side: str, on: bool) -> None:
        default = cfg.LEFT_HAND_COLOR if side == "left" else cfg.RIGHT_HAND_COLOR
        self._vis.set_hand_color(side, cfg.RECORDING_COLOR if on else default)
        self.link.set_recording(side, on)

    def status_line(self):
        """(text, BGR colour) for the OpenCV overlay."""
        parts = [f"{side[0].upper()} REC {rec.frame_count} frames" if rec.recording
                 else f"{side[0].upper()} idle"
                 for side, rec in self.recorders.items()]
        any_rec = any(rec.recording for rec in self.recorders.values())
        return ("Record (thumb+middle pinch): " + "   ".join(parts),
                (0, 0, 255) if any_rec else (150, 150, 150))

    def handle_key(self, key: int) -> bool:
        """L / R replay the newest recording of that hand. Returns True if handled."""
        side = {ord('l'): "left", ord('L'): "left", ord('r'): "right", ord('R'): "right"}.get(key)
        if side is None:
            return False
        path = latest_recording(side)
        if path is None:
            print(f"[Replay] No {side} hand recording in {cfg.RECORDINGS_DIR}")
        else:
            self.link.replay(path)
        return True

    def close(self) -> None:
        for rec in self.recorders.values():   # don't lose a recording still in progress
            if rec.recording:
                rec.stop()
        self.link.close()
