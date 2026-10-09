"""
main_hands_simple.py — Head pose, hand pose and passthrough camera from the Quest,
world frame locked to ArUco marker 100, proximity relock, Open3D visualisation.

Simplified from CoAssembly's main_with_robot.py (no robot, pegboard, board or
secondary markers). All settings live in main_settings.py.

Workflow
--------
  1. Look at marker 100 from within 1 m and roughly face-on — the world frame
     locks automatically the first time (or press ENTER while it is visible).
  2. The world root is streamed to Unity (WORLD_ROOT_PORT) every frame while locked.
  3. To relock later: get within 1 m of marker 100 — the relock cube in Unity
     (Tool Id 100) turns orange — then click it. It flashes green on success.
     ENTER in the OpenCV window also relocks whenever the marker is visible.
  4. Optional (--record-replay): pinch thumb + middle finger to record a hand and
     press L / R to replay it in Unity — see hand_record_replay.py.

Keys (OpenCV window must be focused)
------------------------------------
  ENTER = lock / relock
  L / R = replay the newest left / right hand recording (--record-replay only)
  ESC   = quit

Usage
-----
  python main_hands_simple.py
  python main_hands_simple.py --quest-ip 192.168.50.201
  python main_hands_simple.py --record-replay
"""

import argparse
import json
import struct
import threading
import time

import cv2 as cv
import numpy as np
import open3d as o3d
import zmq
from scipy.spatial.transform import Rotation as ScipyR

import main_settings as cfg


# =============================================================================
# Pose helpers (from CoAssembly utils/pose_helpers.py + utils/unity_conversion.py)
#
# "World" / Open3D frame = Unity frame with Y and Z swapped (Z up, right-handed).
# Camera poses get an extra Rx(-90°) so their axes match OpenCV's camera frame
# (X right, Y down, Z forward) — that lets solvePnP results compose directly.
# =============================================================================

_R_FIX = ScipyR.from_euler('x', -90.0, degrees=True).as_matrix()

# OpenXR hand bones over the 22 flattened joints:
# [0] Wrist, [1] Palm, [2-5] Thumb, [6-9] Index, [10-13] Middle, [14-17] Ring, [18-21] Pinky
_HAND_BONES = np.array([
    [0, 1], [0, 2], [2, 3], [3, 4], [4, 5],
    [0, 6], [6, 7], [7, 8], [8, 9],
    [0, 10], [10, 11], [11, 12], [12, 13],
    [0, 14], [14, 15], [15, 16], [16, 17],
    [0, 18], [18, 19], [19, 20], [20, 21],
], dtype=np.int32)
_N_JOINTS = int(_HAND_BONES.max()) + 1
_JOINT_GROUP_ORDER = ["Wrist", "Palm", "Thumb", "Index", "Middle", "Ring", "Pinky"]
_HIDDEN_PT = np.array([[0., -100., 0.]])


def _hidden_T() -> np.ndarray:
    """Pose used to park geometry out of view instead of removing it."""
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [0., -1.5, 0.]
    return T


def unity_pose_to_T(pos_xyz, rot_xyzw) -> np.ndarray:
    """Unity camera/head pose → 4x4 in the world frame, OpenCV camera axes."""
    x, y, z, w = (float(v) for v in rot_xyzw)
    R = ScipyR.from_quat([-x, -z, -y, w]).as_matrix()   # Unity wxyz → [w, -x, -z, -y]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R @ _R_FIX
    T[:3, 3]  = [float(pos_xyz[0]), float(pos_xyz[2]), float(pos_xyz[1])]
    return T


def T_to_unity(T: np.ndarray):
    """4x4 world-frame transform → (pos, rot_xyzw, column-major 16 floats) in Unity frame."""
    qx, qy, qz, qw = ScipyR.from_matrix(T[:3, :3]).as_quat()
    rot_xyzw = [float(-qx), float(-qz), float(-qy), float(qw)]
    pos = [float(T[0, 3]), float(T[2, 3]), float(T[1, 3])]
    T_unity = np.eye(4, dtype=np.float64)
    T_unity[:3, :3] = ScipyR.from_quat(rot_xyzw).as_matrix()
    T_unity[:3, 3]  = pos
    return pos, rot_xyzw, T_unity.T.flatten().tolist()


def adapt_cx_cy(fx, fy, cx, cy, sensor_w, sensor_h, img_w, img_h):
    """Shift the principal point from full-sensor to the (centre-cropped) image."""
    if sensor_w is None or sensor_h is None:
        return fx, fy, cx, cy
    return (fx, fy,
            cx - (float(sensor_w) - float(img_w)) / 2.0,
            cy - (float(sensor_h) - float(img_h)) / 2.0)


def extract_joints(hand_block) -> "np.ndarray | None":
    """Flatten a Unity hand block into (22, 3) Unity-frame joint positions."""
    if not hand_block:
        return None
    groups = hand_block.get("groups")
    if not groups:
        return None
    joints = []
    for name in _JOINT_GROUP_ORDER:
        for pose in groups.get(name) or []:
            p = (pose or {}).get("position") or {}
            joints.append([p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)])
    return np.array(joints, dtype=np.float64) if joints else None


def to_world(pts_unity: np.ndarray, T_world_tracking: "np.ndarray | None") -> np.ndarray:
    """Unity tracking-space points → world frame (or just axis-swapped if not locked)."""
    pts = pts_unity[:, [0, 2, 1]]
    if T_world_tracking is None:
        return pts
    return pts @ T_world_tracking[:3, :3].T + T_world_tracking[:3, 3]


# =============================================================================
# Unity → Python receivers
# =============================================================================

class CamFeedReceiver:
    """Left passthrough frames from PassthroughCameraPublisher (9-part multipart)."""

    def __init__(self, ip: str, port: int, topic: str):
        self._sub = zmq.Context.instance().socket(zmq.SUB)
        self._sub.setsockopt_string(zmq.SUBSCRIBE, topic)
        self._sub.connect(f"tcp://{ip}:{port}")
        print(f"[Camera] SUB → tcp://{ip}:{port}  topic={topic!r}")

    def poll(self, timeout_ms: int = 0) -> "dict | None":
        """Return the newest frame as a dict, or None if nothing new arrived."""
        if not self._sub.poll(timeout_ms):
            return None
        latest = None
        while True:
            try:
                latest = self._sub.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
        if latest is None or len(latest) != 9:
            return None
        width,  = struct.unpack("<i", latest[2])
        height, = struct.unpack("<i", latest[3])
        pos     = struct.unpack("<fff", latest[4])
        rot     = struct.unpack("<ffff", latest[5])
        fx, fy, cx, cy = struct.unpack("<ffff", latest[6])
        sw, sh  = struct.unpack("<ii", latest[7])
        image = cv.imdecode(np.frombuffer(latest[8], dtype=np.uint8), cv.IMREAD_COLOR)
        if image is None:
            return None
        return {
            "timestamp_ms": struct.unpack("<q", latest[1])[0],
            "image": image,
            "width": width, "height": height,
            "cam_T": unity_pose_to_T(pos, rot),
            "intrinsics": adapt_cx_cy(fx, fy, cx, cy, sw, sh, width, height),
        }

    def close(self):
        self._sub.close(0)


class AnchorDetector:
    """Detects the anchor marker and returns T_cam_anchor (OpenCV camera frame)."""

    def __init__(self, marker_id: int, marker_size_m: float, dictionary_name: str):
        self.marker_id = int(marker_id)
        self.size = float(marker_size_m)
        dictionary = cv.aruco.getPredefinedDictionary(getattr(cv.aruco, dictionary_name))
        self._detector = cv.aruco.ArucoDetector(dictionary, cv.aruco.DetectorParameters())
        s = self.size / 2.0
        self._obj_pts = np.array([[-s, s, 0.], [s, s, 0.], [s, -s, 0.], [-s, -s, 0.]],
                                 dtype=np.float64)

    def detect(self, bgr, fx, fy, cx, cy):
        """Return (T_cam_anchor or None, annotated image)."""
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros((5, 1))
        corners, ids, _ = self._detector.detectMarkers(cv.cvtColor(bgr, cv.COLOR_BGR2GRAY))
        vis = bgr.copy()
        if ids is None:
            return None, vis
        cv.aruco.drawDetectedMarkers(vis, corners, ids)
        for c, mid in zip(corners, ids.flatten()):
            if int(mid) != self.marker_id:
                continue
            ok, rvec, tvec = cv.solvePnP(self._obj_pts, c.reshape(4, 2).astype(np.float64),
                                         K, dist, flags=cv.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            cv.drawFrameAxes(vis, K, dist, rvec, tvec, self.size * 0.5, 2)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = cv.Rodrigues(rvec)[0]
            T[:3, 3]  = tvec.reshape(3)
            return T, vis
        return None, vis


class ArUcoWorker:
    """Polls the camera and runs detection on a background thread so the main
    loop never blocks on JPEG decode / marker detection (~15-20 ms).

    Each result keeps the camera pose from the same frame as the detection, so
    a lock always pairs T_cam_anchor with the camera pose it was measured from.
    """

    def __init__(self, cam: CamFeedReceiver, detector: AnchorDetector):
        self._cam = cam
        self._detector = detector
        self._lock = threading.Lock()
        self._latest = None
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            frame = self._cam.poll(timeout_ms=10)
            if frame is None:
                continue
            T_cam_anchor, vis = self._detector.detect(frame["image"], *frame["intrinsics"])
            frame["T_cam_anchor"] = T_cam_anchor
            frame["det_vis"] = vis
            with self._lock:
                self._latest = frame

    def get(self) -> "dict | None":
        with self._lock:
            return self._latest

    def stop(self):
        self._running = False
        self._thread.join(timeout=1.0)


class HandDataReceiver:
    """Head + hand JSON from TrackingDataManager / HandTrackingSenderNetMQ."""

    def __init__(self, ip: str, port: int):
        self._sub = zmq.Context.instance().socket(zmq.SUB)
        self._sub.setsockopt(zmq.CONFLATE, 1)
        self._sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sub.connect(f"tcp://{ip}:{port}")
        self.data = None
        self.last_rx_time = None
        print(f"[Hands] SUB → tcp://{ip}:{port}")

    def poll(self) -> bool:
        latest = None
        while True:
            try:
                latest = self._sub.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
        if latest is None:
            return False
        try:
            self.data = json.loads(latest)
        except json.JSONDecodeError as e:
            print(f"[Hands] Bad JSON: {e}")
            return False
        self.last_rx_time = time.time()
        return True

    @property
    def receiving(self) -> bool:
        return self.last_rx_time is not None and time.time() - self.last_rx_time < 2.0

    def _center_eye(self) -> "dict | None":
        if self.data is None:
            return None
        return (self.data.get("head") or {}).get("CenterEye")

    def center_eye_T(self) -> "np.ndarray | None":
        """Head pose in tracking space (world-frame axes, OpenCV camera convention)."""
        ce = self._center_eye()
        if not ce:
            return None
        p = ce.get("position") or {}
        r = ce.get("rotation") or {}
        return unity_pose_to_T([p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)],
                               [r.get("x", 0.0), r.get("y", 0.0), r.get("z", 0.0), r.get("w", 1.0)])

    def world_joints(self, T_world_tracking: "np.ndarray | None"):
        """Return (left, right) as (22, 3) world-frame arrays, or None per hand."""
        if self.data is None:
            return None, None
        hands = self.data.get("hands") or {}

        head_world = None
        ce = self._center_eye()
        if ce:
            p = ce.get("position") or {}
            head_world = to_world(np.array([[p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)]]),
                                  T_world_tracking)[0]

        def _resolve(key):
            joints = extract_joints(hands.get(key))
            if joints is None:
                return None
            pts = to_world(joints, T_world_tracking)
            if (head_world is not None
                    and np.linalg.norm(pts[1] - head_world) > cfg.MAX_HAND_HEAD_DIST):
                return None
            return pts

        return _resolve("LeftHand"), _resolve("RightHand")

    def pinch_strength(self, key: str) -> "float | None":
        block = self.hand_block(key)
        return block.get("indexPinchStrength") if block else None

    def hand_block(self, key: str) -> "dict | None":
        """Raw JSON block for 'LeftHand' / 'RightHand' (None if not tracked)."""
        if self.data is None:
            return None
        return (self.data.get("hands") or {}).get(key)

    @property
    def unity_timestamp(self) -> "float | None":
        """Unity's Time.time when the current message was captured."""
        return self.data.get("timestamp") if self.data else None

    def close(self):
        self._sub.close(0)


# =============================================================================
# Python → Unity
# =============================================================================

class WorldAnchor:
    """Holds T_world_tracking (tracking space → marker-100 world) and streams
    its inverse to Unity's WorldRootPoseReceiverNetMQ."""

    def __init__(self, ip: str, port: int):
        self._T_wt: "np.ndarray | None" = None
        self._pub = zmq.Context.instance().socket(zmq.PUB)
        self._pub.connect(f"tcp://{ip}:{port}")
        print(f"[WorldRoot] PUB → tcp://{ip}:{port}")

    @property
    def locked(self) -> bool:
        return self._T_wt is not None

    @property
    def T_world_tracking(self) -> "np.ndarray | None":
        return self._T_wt

    def lock(self, T_cam_anchor: np.ndarray, cam_T: np.ndarray) -> bool:
        if T_cam_anchor is None or cam_T is None:
            return False
        self._T_wt = np.linalg.inv(cam_T @ T_cam_anchor)
        return True

    def world_T(self, T_tracking: "np.ndarray | None") -> "np.ndarray | None":
        if self._T_wt is None or T_tracking is None:
            return None
        return self._T_wt @ T_tracking

    def publish(self) -> None:
        if self._T_wt is None:
            return
        pos, rot, mat = T_to_unity(np.linalg.inv(self._T_wt))
        self._pub.send_string(json.dumps({
            "world_root_position": pos,
            "world_root_rotation_xyzw": rot,
            "world_root_matrix": mat,
        }))

    def close(self):
        self._pub.close(0)


class RelockButton:
    """Unity relock cube: click events in (ToolClickPublisher), colour out (ToolColorReceiver)."""

    def __init__(self, ip: str, click_port: int, color_port: int, tool_id: int):
        ctx = zmq.Context.instance()
        self.tool_id = int(tool_id)
        self._sub = ctx.socket(zmq.SUB)
        self._sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sub.connect(f"tcp://{ip}:{click_port}")
        self._pub = ctx.socket(zmq.PUB)
        self._pub.connect(f"tcp://{ip}:{color_port}")
        self._color = cfg.RESET_COLOR
        self._last_color_send = 0.0
        self._clicked = False
        print(f"[Relock] SUB → tcp://{ip}:{click_port}  PUB → tcp://{ip}:{color_port}  "
              f"tool_id={self.tool_id}")

    def poll(self) -> None:
        while True:
            try:
                raw = self._sub.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                msg = json.loads(raw)
                tool_id = int(msg["tool_id"])
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"[Relock] Bad click message: {e}")
                continue
            if tool_id == self.tool_id and msg.get("event_type") in ("selected", "long_press"):
                self._clicked = True
        # PUB/SUB keeps no state — repeat the current colour so a Unity
        # subscriber that connects late still ends up showing the right one.
        if time.monotonic() - self._last_color_send >= cfg.COLOR_REFRESH_S:
            self._send(self._color)

    def consume_click(self) -> bool:
        clicked, self._clicked = self._clicked, False
        return clicked

    def set_color(self, color) -> None:
        if color != self._color:
            self._color = color
            self._send(color)

    def _send(self, color) -> None:
        self._pub.send_string(json.dumps({"tool_id": self.tool_id,
                                          "color": [float(c) for c in color]}))
        self._last_color_send = time.monotonic()

    def close(self):
        self._sub.close(0)
        self._pub.close(200)   # brief linger so the final RESET colour reaches Unity


# =============================================================================
# Open3D scene
# =============================================================================

class SceneVis:
    """World axes, tracking origin, passthrough-camera frustum, head frustum and
    both hand skeletons. Unused geometry is parked at _hidden_T()."""

    def __init__(self, title: str):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(title, width=1000, height=680)
        ro = self.vis.get_render_option()
        ro.background_color = np.array([0.08, 0.08, 0.10])
        ro.point_size = 7.0
        ro.line_width = 2.0

        self.vis.add_geometry(
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=cfg.WORLD_FRAME_SIZE))

        self._tracking_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)
        self._tracking_frame.transform(_hidden_T())
        self.vis.add_geometry(self._tracking_frame)
        self._tracking_T = _hidden_T()

        self._frustum_colors = {}
        self._cam_frustum  = self._add_frustum([1.0, 1.0, 0.0])   # yellow: passthrough camera
        self._head_frustum = self._add_frustum([1.0, 0.1, 0.9])   # magenta: CenterEye

        self._left  = self._add_hand(cfg.LEFT_HAND_COLOR)
        self._right = self._add_hand(cfg.RIGHT_HAND_COLOR)

        ctr = self.vis.get_view_control()
        ctr.set_lookat([0., 0., 0.])
        ctr.set_front([0., -0.5, -1.])
        ctr.set_up([0., 1., 0.])
        ctr.set_zoom(0.5)

    @staticmethod
    def _frustum_lines(T, w, h, fx, fy, cx, cy) -> o3d.geometry.LineSet:
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        return o3d.geometry.LineSet.create_camera_visualization(
            int(w), int(h), K, np.linalg.inv(T), scale=cfg.FRUSTUM_SCALE)

    def _add_frustum(self, color) -> o3d.geometry.LineSet:
        ls = self._frustum_lines(_hidden_T(), 640, 480, 400., 400., 320., 240.)
        ls.paint_uniform_color(color)
        self._frustum_colors[id(ls)] = color
        self.vis.add_geometry(ls)
        return ls

    def _set_frustum(self, ls, T, w=640, h=480, fx=400., fy=400., cx=320., cy=240.):
        new = self._frustum_lines(T if T is not None else _hidden_T(), w, h, fx, fy, cx, cy)
        ls.points = new.points
        ls.lines  = new.lines
        ls.paint_uniform_color(self._frustum_colors[id(ls)])
        self.vis.update_geometry(ls)

    def _add_hand(self, color):
        dummy = np.tile(_HIDDEN_PT, (_N_JOINTS, 1))
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dummy))
        pcd.paint_uniform_color(color)
        lines = o3d.geometry.LineSet(o3d.utility.Vector3dVector(dummy),
                                     o3d.utility.Vector2iVector(_HAND_BONES))
        lines.paint_uniform_color(color)
        self.vis.add_geometry(pcd)
        self.vis.add_geometry(lines)
        return pcd, lines

    def _set_hand(self, hand, pts):
        pcd, lines = hand
        if pts is None or len(pts) == 0:
            use = np.tile(_HIDDEN_PT, (_N_JOINTS, 1))
        else:
            use = np.zeros((_N_JOINTS, 3))
            m = min(len(pts), _N_JOINTS)
            use[:m] = pts[:m]
        pcd.points   = o3d.utility.Vector3dVector(use)
        lines.points = o3d.utility.Vector3dVector(use)
        self.vis.update_geometry(pcd)
        self.vis.update_geometry(lines)

    def update_tracking(self, T):
        T_new = T if T is not None else _hidden_T()
        self._tracking_frame.transform(T_new @ np.linalg.inv(self._tracking_T))
        self._tracking_T = T_new
        self.vis.update_geometry(self._tracking_frame)

    def update_camera(self, T, w, h, fx, fy, cx, cy):
        self._set_frustum(self._cam_frustum, T, w, h, fx, fy, cx, cy)

    def update_head(self, T):
        self._set_frustum(self._head_frustum, T)

    def update_hands(self, left, right):
        self._set_hand(self._left, left)
        self._set_hand(self._right, right)

    def set_hand_color(self, side: str, color) -> None:
        pcd, lines = self._left if side == "left" else self._right
        pcd.paint_uniform_color(color)
        lines.paint_uniform_color(color)
        self.vis.update_geometry(pcd)
        self.vis.update_geometry(lines)

    def tick(self) -> bool:
        """Pump the window; returns False once the user closed it."""
        alive = self.vis.poll_events()
        self.vis.update_renderer()
        return alive

    def close(self):
        self.vis.destroy_window()


# =============================================================================
# Main scene
# =============================================================================

class MainScene:
    def __init__(self, quest_ip: str, record_replay: bool = False):
        self.cam    = CamFeedReceiver(quest_ip, cfg.CAM_FEED_PORT, cfg.CAM_TOPIC)
        self.aruco  = ArUcoWorker(self.cam, AnchorDetector(cfg.ANCHOR_MARKER_ID,
                                                           cfg.ANCHOR_MARKER_SIZE,
                                                           cfg.ARUCO_DICT))
        self.hands  = HandDataReceiver(quest_ip, cfg.HAND_PORT)
        self.anchor = WorldAnchor(quest_ip, cfg.WORLD_ROOT_PORT)
        self.relock = RelockButton(quest_ip, cfg.TOOL_CLICK_PORT, cfg.TOOL_COLOR_PORT,
                                   cfg.RELOCK_TOOL_ID)
        self.vis    = SceneVis(f"Hands Simple — world frame (marker #{cfg.ANCHOR_MARKER_ID})")

        # Optional hand record / replay add-on (--record-replay).
        self.record_replay = None
        if record_replay:
            from hand_record_replay import HandRecordReplay
            self.record_replay = HandRecordReplay(quest_ip, self.vis)

        extra_keys = self.record_replay.KEYS if self.record_replay else ""
        self._win   = (f"Quest Left Passthrough  [ENTER=lock/relock{extra_keys}  ESC=quit]"
                       f"  anchor=#{cfg.ANCHOR_MARKER_ID}")
        cv.namedWindow(self._win, cv.WINDOW_NORMAL)
        cv.resizeWindow(self._win, *cfg.CV_WINDOW_SIZE)

        self._last_relock_time = 0.0
        self._flash_until = 0.0

    # ── Locking ───────────────────────────────────────────────────────────────

    def _lock(self, frame: dict, reason: str) -> None:
        was_locked = self.anchor.locked
        if not self.anchor.lock(frame["T_cam_anchor"], frame["cam_T"]):
            return
        self._last_relock_time = time.time()
        dist = float(np.linalg.norm(frame["T_cam_anchor"][:3, 3]))
        print(f"[{reason}] {'Relocked' if was_locked else 'Locked'} world to marker "
              f"#{cfg.ANCHOR_MARKER_ID} ({dist:.2f} m)")

    def _update_locking(self, frame: "dict | None", head_ok: bool) -> None:
        now = time.time()
        T_cam_anchor = frame["T_cam_anchor"] if frame else None
        dist = (float(np.linalg.norm(T_cam_anchor[:3, 3]))
                if T_cam_anchor is not None else float("inf"))

        # Auto-lock on first sight only. T_cam_anchor[2, 3] / dist is the cosine
        # of the angle between the camera's optical axis and the marker direction,
        # so it rejects oblique, edge-of-view detections.
        if not self.anchor.locked and T_cam_anchor is not None and head_ok:
            cos_tilt = T_cam_anchor[2, 3] / dist if dist > 1e-6 else 0.0
            if (dist < cfg.AUTO_LOCK_MAX_DIST
                    and cos_tilt > np.cos(np.deg2rad(cfg.AUTO_LOCK_MAX_TILT_DEG))):
                self._lock(frame, "AutoLock")

        # Proximity relock: the cube is orange while a relock is possible,
        # green for RELOCK_FLASH_S after one, otherwise its original colour.
        available = (self.anchor.locked and T_cam_anchor is not None
                     and dist < cfg.RELOCK_MAX_DIST)
        clicked = self.relock.consume_click()
        if (clicked and available
                and now - self._last_relock_time >= cfg.RELOCK_COOLDOWN):
            self._lock(frame, "ProximityRelock")
            self._flash_until = now + cfg.RELOCK_FLASH_S
        elif clicked and not available:
            print(f"[ProximityRelock] Click ignored — marker #{cfg.ANCHOR_MARKER_ID} "
                  f"not visible within {cfg.RELOCK_MAX_DIST:.1f} m")

        if now < self._flash_until:
            self.relock.set_color(cfg.SELECTED_COLOR)
        else:
            self.relock.set_color(cfg.AVAILABLE_COLOR if available else cfg.RESET_COLOR)

    # ── OpenCV overlay ────────────────────────────────────────────────────────

    def _draw_status(self, frame, left, right) -> None:
        img = frame["det_vis"] if frame else np.zeros((480, 640, 3), dtype=np.uint8)
        disp = cv.resize(img, cfg.CV_WINDOW_SIZE)
        marker_ok = frame is not None and frame["T_cam_anchor"] is not None
        green, grey, red = (0, 255, 80), (150, 150, 150), (0, 80, 255)

        lines = [
            (f"Marker #{cfg.ANCHOR_MARKER_ID}: "
             + (f"DETECTED {np.linalg.norm(frame['T_cam_anchor'][:3, 3]):.2f} m"
                if marker_ok else "searching..."),
             green if marker_ok else red),
            (f"World: {'LOCKED -> Unity :' + str(cfg.WORLD_ROOT_PORT) if self.anchor.locked else 'not locked'}",
             green if self.anchor.locked else grey),
            (f"Camera: {'receiving' if frame else 'waiting for :' + str(cfg.CAM_FEED_PORT)}",
             green if frame else red),
        ]
        if self.hands.receiving:
            hand_names = [n for n, p in (("L", left), ("R", right)) if p is not None]
            pinches = []
            for key in ("LeftHand", "RightHand"):
                strength = self.hands.pinch_strength(key)
                if strength is not None:
                    pinches.append(f"{key[0]} pinch {strength:.2f}")
            lines.append((f"Head: OK   Hands: {'+'.join(hand_names) or 'none'}   "
                          f"{'  '.join(pinches)}", green))
        else:
            lines.append((f"Head/hands: waiting for :{cfg.HAND_PORT}", red))

        if self.record_replay:
            lines.append(self.record_replay.status_line())

        for i, (text, color) in enumerate(lines):
            cv.putText(disp, text, (12, 34 + 32 * i), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv.imshow(self._win, disp)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        try:
            while True:
                self.hands.poll()
                self.relock.poll()
                frame = self.aruco.get()
                center_T = self.hands.center_eye_T()

                self._update_locking(frame, head_ok=center_T is not None)
                self.anchor.publish()

                T_wt = self.anchor.T_world_tracking
                left, right = self.hands.world_joints(T_wt)
                if self.record_replay:
                    self.record_replay.update(self.hands, left, right, T_wt)

                self.vis.update_tracking(T_wt)
                self.vis.update_head(self.anchor.world_T(center_T))
                self.vis.update_hands(left, right)
                if frame is not None:
                    self.vis.update_camera(self.anchor.world_T(frame["cam_T"]),
                                           frame["width"], frame["height"],
                                           *frame["intrinsics"])
                if not self.vis.tick():
                    break

                self._draw_status(frame, left, right)
                key = cv.waitKey(1) & 0xFF
                if key == 27:   # ESC
                    break
                if key == 13:   # ENTER
                    if frame is None or frame["T_cam_anchor"] is None:
                        print(f"[ENTER] Marker #{cfg.ANCHOR_MARKER_ID} not visible — cannot lock.")
                    else:
                        self._lock(frame, "ENTER")
                elif self.record_replay:
                    self.record_replay.handle_key(key)
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self) -> None:
        self.aruco.stop()
        self.relock.set_color(cfg.RESET_COLOR)
        parts = [self.cam, self.hands, self.anchor, self.relock]
        if self.record_replay:
            parts.append(self.record_replay)   # also saves any recording still in progress
        for part in parts:
            try:
                part.close()
            except Exception:
                pass
        self.vis.close()
        cv.destroyAllWindows()
        zmq.Context.instance().term()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quest-ip", default=cfg.UNITY_IP,
                        help=f"IP of the Quest running Unity (default {cfg.UNITY_IP})")
    parser.add_argument("--record-replay", action="store_true",
                        help="enable pinch-to-record hands and L/R replay in Unity "
                             "(hand_record_replay.py)")
    args = parser.parse_args()
    MainScene(args.quest_ip, record_replay=args.record_replay).run()


if __name__ == "__main__":
    main()
