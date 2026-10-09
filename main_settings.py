# main_settings.py
#
# Single source of truth for IPs, ports, ArUco settings and relock thresholds
# used by main_hands_simple.py. Edit values here — no need to touch the main script.
#
# Socket convention (same as CoAssembly): Unity BINDS every socket, Python CONNECTS
# to UNITY_IP. Ports must match the Unity components listed next to each one.

# ── Machine IPs ───────────────────────────────────────────────────────────────
UNITY_IP = "192.168.50.167"   # Quest running the Unity app (override with --quest-ip)

# ── Ports (Unity → Python) ────────────────────────────────────────────────────
CAM_FEED_PORT   = 6560   # left passthrough JPEG + pose + intrinsics (PassthroughCameraPublisher)
HAND_PORT       = 6570   # head + hand tracking JSON (TrackingDataManager → HandTrackingSenderNetMQ)
TOOL_CLICK_PORT = 6001   # select / hover events (ToolClickPublisher)

# ── Ports (Python → Unity) ────────────────────────────────────────────────────
WORLD_ROOT_PORT = 6000   # world root transform (WorldRootPoseReceiverNetMQ)
TOOL_COLOR_PORT = 6002   # per-tool highlight colour (ToolColorReceiver)

# ── ArUco ─────────────────────────────────────────────────────────────────────
ARUCO_DICT         = "DICT_6X6_1000"   # name of a cv2.aruco predefined dictionary
ANCHOR_MARKER_ID   = 100               # world-frame anchor
ANCHOR_MARKER_SIZE = 0.100             # metres (black square edge length)
CAM_TOPIC          = "cam_left"        # PassthroughCameraPublisher topic to subscribe to

# ── Auto-lock (first lock only) ───────────────────────────────────────────────
AUTO_LOCK_MAX_DIST     = 1.0    # metres — marker must be closer than this
AUTO_LOCK_MAX_TILT_DEG = 45.0   # max angle between camera optical axis and marker direction

# ── Proximity relock (click the relock cube in Unity) ─────────────────────────
RELOCK_TOOL_ID   = ANCHOR_MARKER_ID   # ToolClickPublisher / ToolColorReceiver Tool Id on the relock cube
RELOCK_MAX_DIST  = 1.0                # metres — cube is clickable only within this range
RELOCK_COOLDOWN  = 2.0                # seconds between relocks
RELOCK_FLASH_S   = 1.0                # seconds the cube stays green after a relock

# Relock cube colours sent on TOOL_COLOR_PORT (RGBA, 0-1).
AVAILABLE_COLOR = [1.0, 0.5, 0.0, 0.75]     # orange — relock available
SELECTED_COLOR  = [0.0, 1.0, 0.0, 0.75]     # green  — just relocked
RESET_COLOR     = [-1.0, -1.0, -1.0, -1.0]  # sentinel → Unity restores the original colour
COLOR_REFRESH_S = 1.0                       # republish current colour so late subscribers catch up

# ── Hand filtering ────────────────────────────────────────────────────────────
MAX_HAND_HEAD_DIST = 2.0   # metres — drop a hand whose palm is farther than this from the head

# ── Visualisation ─────────────────────────────────────────────────────────────
WORLD_FRAME_SIZE  = 0.30            # Open3D world axes length (m)
FRUSTUM_SCALE     = 0.20            # Open3D camera/head frustum depth (m)
LEFT_HAND_COLOR   = [0.3, 0.6, 1.0]
RIGHT_HAND_COLOR  = [1.0, 0.55, 0.1]
CV_WINDOW_SIZE    = (960, 540)      # OpenCV passthrough window (w, h)
