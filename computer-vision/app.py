# Library imports
import argparse
import asyncio
import base64
import json
import os
import queue
import threading
import time
import urllib.request

from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# -----------------------------------------------------------------------------
# CLI ARGUMENTS
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0, help="Camera device number")
parser.add_argument("--width", type=int, default=960, help="Camera capture width")
parser.add_argument("--height", type=int, default=540, help="Camera capture height")
parser.add_argument("--use_static_image_mode", action="store_true",
                    help="Use static image mode for MediaPipe")
parser.add_argument("--min_detection_confidence", type=float, default=0.5,
                    help="Detection confidence threshold")
parser.add_argument("--min_tracking_confidence", type=float, default=0.5,
                    help="Tracking confidence threshold")
parser.add_argument("--server_url", type=str,
                    default="ws://127.0.0.1:8000/ws/ingest",
                    help="server-control ingest websocket URL")
parser.add_argument("--no_server", action="store_true",
                    help="Disable pushing frames to server-control")
parser.add_argument("--smoothing_ms", type=int, default=200,
                    help="Rolling-window length (ms) for smoothing angle/openness signals")
args = parser.parse_args()

# -----------------------------------------------------------------------------
# MODEL DOWNLOAD (Tasks API needs .task files on disk)
# -----------------------------------------------------------------------------

# Models are fetched once and cached next to app.py in ./models/
MODELS = {
    "pose_landmarker_lite.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}

def ensure_models(models_dir):
    """Download any missing .task model files into models_dir."""
    os.makedirs(models_dir, exist_ok=True)
    for name, url in MODELS.items():
        path = os.path.join(models_dir, name)
        if not os.path.exists(path):
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(url, path)
            print(f"  saved to {path}")
    return models_dir

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = ensure_models(os.path.join(SCRIPT_DIR, "models"))

# -----------------------------------------------------------------------------
# LANDMARK INDICES
# -----------------------------------------------------------------------------

# BlazePose full-body indices for the 6 upper-body joints we care about.
# These indices are stable across MediaPipe releases.
POSE_IDX = {
    "LEFT_ELBOW":  13,
    "RIGHT_ELBOW": 14,
    "LEFT_WRIST":  15,
    "RIGHT_WRIST": 16,
}

# MediaPipe Hands 21-landmark model indices
HAND_WRIST = 0
INDEX_TIP, INDEX_PIP   = 8, 6
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP, RING_PIP     = 16, 14
PINKY_TIP, PINKY_PIP   = 20, 18
MIDDLE_FINGER_MCP = 9

FINGER_TIPS = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_PIPS = [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]

# Forearm yaw: horizontal sweep derived from elbow->wrist x/z. Clamped to the
# base yaw servo's mechanical range. Monocular depth (MediaPipe .z) is noisy,
# so smoothing happens downstream via the shared rolling-window smoother.
YAW_LIMIT_DEG = 80.0

# Hand openness: linear map from avg wrist-tip/wrist-pip ratio → [0, 1].
# Thumb is deliberately excluded because it moves laterally and biases the score.
OPENNESS_MIN_RATIO = 0.65   # ratio at which we call it 0%
OPENNESS_MAX_RATIO = 1.35   # ratio at which we call it 100%

# Hand-open hysteresis. Two-sided band kills the boundary flicker you get
# from a single threshold when the smoothed pct sits right at 50%.
OPEN_ENTER = 0.55   # closed → open when smoothed pct crosses upward
OPEN_EXIT  = 0.45   # open → closed when smoothed pct crosses downward
_hand_is_open_state = False

# Annotated-frame streaming to the dashboard. JPEG encode + base64 is expensive,
# so we cap the cadence — the stats table still updates at full CV rate.
DASHBOARD_FPS = 15
_last_frame_send_ms = 0

# Edge list for drawing the 21-landmark hand skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # palm base
]

# -----------------------------------------------------------------------------
# HAND OPENNESS
# -----------------------------------------------------------------------------

def hand_openness(lm):
    """
    Continuous openness score in [0.0, 1.0] based on how far each fingertip
    is from the wrist relative to its PIP joint. Scale-invariant (pure ratio).
    Thumb is excluded — it moves laterally and biases the score.
    Returns pct. Open/closed state is derived downstream from the smoothed
    pct with hysteresis.
    """
    wrist = np.array([lm[HAND_WRIST].x, lm[HAND_WRIST].y])
    ratios = []
    for tip_idx, pip_idx in zip(FINGER_TIPS, FINGER_PIPS):
        tip = np.array([lm[tip_idx].x, lm[tip_idx].y])
        pip = np.array([lm[pip_idx].x, lm[pip_idx].y])
        d_tip = np.linalg.norm(tip - wrist)
        d_pip = np.linalg.norm(pip - wrist)
        if d_pip < 1e-6:
            continue
        ratios.append(d_tip / d_pip)

    if not ratios:
        return 0.0

    avg = sum(ratios) / len(ratios)
    pct = (avg - OPENNESS_MIN_RATIO) / (OPENNESS_MAX_RATIO - OPENNESS_MIN_RATIO)
    return max(0.0, min(1.0, pct))


# -----------------------------------------------------------------------------
# ROLLING-WINDOW SMOOTHER
# -----------------------------------------------------------------------------

class WindowedMean:
    """Rolling mean over a fixed time window in milliseconds.

    Samples outside the window are dropped on each `add`, so if a signal
    goes invisible for longer than the window the smoother naturally
    drains and does not emit stale pre-occlusion data.
    """

    def __init__(self, window_ms):
        self.window_ms = window_ms
        self.samples: "deque[tuple[int, float]]" = deque()

    def add(self, ts_ms, value):
        self.samples.append((ts_ms, value))
        cutoff = ts_ms - self.window_ms
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

    def value(self):
        if not self.samples:
            return None
        return sum(v for _, v in self.samples) / len(self.samples)


elev_smoother  = WindowedMean(args.smoothing_ms)
yaw_smoother   = WindowedMean(args.smoothing_ms)
wrist_smoother = WindowedMean(args.smoothing_ms)
hand_smoother  = WindowedMean(args.smoothing_ms)


# -----------------------------------------------------------------------------
# FUNCTIONS TO CALCULATE ANGLES
# -----------------------------------------------------------------------------

# Returns the angle in degrees at point b, formed by the vectors b->a and b->c
def angle_between(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

# Returns the angle of a vector relative to horizontal (0° = flat, 90° = pointing up)
# Negative y because image coords have y increasing downward
def elevation_angle(p1, p2):
    vec = np.array(p2) - np.array(p1)
    return np.degrees(np.arctan2(-vec[1], abs(vec[0])))

# -----------------------------------------------------------------------------
# HAND SKELETON DRAWING (replaces mp.solutions.drawing_utils)
# -----------------------------------------------------------------------------

def draw_hand_skeleton(frame, hand_landmarks, color):
    """Draw the 21-landmark hand skeleton using plain OpenCV primitives."""
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in hand_landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for p in pts:
        cv2.circle(frame, p, 3, color, -1)

# -----------------------------------------------------------------------------
# FRAME PAYLOAD + WEBSOCKET SENDER
# -----------------------------------------------------------------------------

def build_frame_payload(ts_ms, forearm_elev, forearm_yaw, wrist_bend,
                        hand_side, hand_pct, hand_is_open, frame_b64=None):
    """Assemble the JSON payload for one CV frame (see server-control/README.md)."""
    forearm = {"visible": forearm_elev is not None}
    if forearm_elev is not None:
        forearm["elevation_deg"] = round(float(forearm_elev), 2)
    if forearm_yaw is not None:
        forearm["yaw_deg"] = round(float(forearm_yaw), 2)

    wrist = {"visible": wrist_bend is not None}
    if wrist_bend is not None:
        wrist["bend_deg"] = round(float(wrist_bend), 2)

    hand = {"visible": hand_side is not None}
    if hand_side is not None:
        hand["side"] = hand_side
        hand["openness_pct"] = float(hand_pct)
        hand["is_open"] = bool(hand_is_open)

    payload = {
        "ts_ms": int(ts_ms),
        "forearm": forearm,
        "wrist": wrist,
        "hand": hand,
    }
    if frame_b64 is not None:
        payload["frame_jpeg_b64"] = frame_b64
    return payload


class FramePusher:
    """
    Pushes frame payloads to server-control over a websocket.

    OpenCV's main loop stays synchronous — we run asyncio + the websockets
    library on a dedicated background thread and communicate through a
    size-1 queue. If the queue is full we drop the oldest frame: for a live
    feed, freshness beats completeness.
    """

    _SENTINEL = object()

    def __init__(self, url):
        self.url = url
        self.q: "queue.Queue" = queue.Queue(maxsize=1)
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def push(self, payload: dict) -> None:
        # Drop stale frame if the sender hasn't caught up yet.
        try:
            self.q.put_nowait(payload)
        except queue.Full:
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(payload)
            except queue.Full:
                pass

    def stop(self):
        try:
            self.q.put_nowait(self._SENTINEL)
        except queue.Full:
            # Replace whatever is queued with the sentinel.
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(self._SENTINEL)
            except queue.Full:
                pass
        self.thread.join(timeout=2.0)

    def _run(self):
        try:
            asyncio.run(self._sender_loop())
        except Exception as e:
            print(f"[server push] sender thread exited: {e}")

    async def _sender_loop(self):
        # Imported lazily so users running with --no_server don't need the dep.
        import websockets

        loop = asyncio.get_running_loop()

        while True:
            try:
                async with websockets.connect(self.url, max_queue=1) as ws:
                    print(f"[server push] connected to {self.url}")
                    while True:
                        item = await loop.run_in_executor(None, self.q.get)
                        if item is self._SENTINEL:
                            return
                        await ws.send(json.dumps(item))
            except asyncio.CancelledError:
                return
            except Exception as e:
                print(f"[server push] connection failed ({e}); retrying in 1s")
                # Drain any queued frames so we don't send stale data on reconnect.
                try:
                    while True:
                        item = self.q.get_nowait()
                        if item is self._SENTINEL:
                            return
                except queue.Empty:
                    pass
                await asyncio.sleep(1.0)


pusher = None
if not args.no_server:
    pusher = FramePusher(args.server_url)
    pusher.start()

# -----------------------------------------------------------------------------
# INITIALISATION OF TASK API LANDMARKERS AND WEBCAM
# -----------------------------------------------------------------------------

BaseOptions = mp_python.BaseOptions
VisionRunningMode = mp_vision.RunningMode

running_mode = (
    VisionRunningMode.IMAGE if args.use_static_image_mode else VisionRunningMode.VIDEO
)

pose_options = mp_vision.PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=os.path.join(MODELS_DIR, "pose_landmarker_lite.task")
    ),
    running_mode=running_mode,
    num_poses=1,
    min_pose_detection_confidence=args.min_detection_confidence,
    min_pose_presence_confidence=args.min_detection_confidence,
    min_tracking_confidence=args.min_tracking_confidence,
)
pose_landmarker = mp_vision.PoseLandmarker.create_from_options(pose_options)

hand_options = mp_vision.HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=os.path.join(MODELS_DIR, "hand_landmarker.task")
    ),
    running_mode=running_mode,
    num_hands=1,
    min_hand_detection_confidence=args.min_detection_confidence,
    min_hand_presence_confidence=args.min_tracking_confidence,
    min_tracking_confidence=args.min_tracking_confidence,
)
hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)

cap = cv2.VideoCapture(args.device)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open camera device {args.device}. "
        f"Try a different --device index (0, 1, 2, ...)."
    )


# -----------------------------------------------------------------------------
# THREADED FRAME GRABBER — eliminates camera buffer lag
# -----------------------------------------------------------------------------

class FrameGrabber:
    """Continuously grabs frames in a background thread, keeping only the latest."""
    def __init__(self, cap):
        self.cap = cap
        self._frame = None
        self._ret = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()

    def _grab_loop(self):
        while True:
            ret, frame = self.cap.read()
            with self._lock:
                self._ret = ret
                self._frame = frame
            if not ret:
                break

    def read(self):
        with self._lock:
            return self._ret, self._frame.copy() if self._frame is not None else None


grabber = FrameGrabber(cap)

# -----------------------------------------------------------------------------
# RUN MODELS WHILE WEBCAM IS RUNNING
# -----------------------------------------------------------------------------
start_ns = time.monotonic_ns()

while cap.isOpened():
    ret, frame = grabber.read()
    if not ret:
        break

    # Flip so the view mirrors the user (more intuitive for self-monitoring)
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # Monotonic frame clock — used both as MediaPipe's VIDEO-mode timestamp and
    # as the key into our rolling-window smoothers.
    current_ms = (time.monotonic_ns() - start_ns) // 1_000_000

    # MediaPipe Tasks API expects an mp.Image wrapping an RGB ndarray
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Run both landmarkers on the same frame
    if running_mode == VisionRunningMode.VIDEO:
        pose_results = pose_landmarker.detect_for_video(mp_image, current_ms)
        hands_results = hand_landmarker.detect_for_video(mp_image, current_ms)
    else:
        pose_results = pose_landmarker.detect(mp_image)
        hands_results = hand_landmarker.detect(mp_image)

    # Store pose wrist positions so the hand section can look them up:
    # { "LEFT": (x, y), "RIGHT": (x, y) } in normalised coords
    pose_wrist_positions = {}
    pose_lm = None  # populated below if a pose was detected

    # Frame-scope values the payload builder will read after rendering.
    # Stay None if the corresponding signal wasn't visible this frame.
    frame_forearm_elev = None
    frame_forearm_yaw  = None
    frame_wrist_bend   = None
    frame_hand_side    = None
    frame_hand_pct     = None
    frame_hand_open    = None

    # -----------------------------------------------------------------------------
    # POSE RENDERING
    # -----------------------------------------------------------------------------
    if pose_results.pose_landmarks:
        # Tasks API returns a list of pose instances; we only track one person
        pose_lm = pose_results.pose_landmarks[0]

        # We only want the user's RIGHT arm. Because we mirror the frame with
        # cv2.flip(frame, 1) before inference, MediaPipe sees a flipped person
        # and labels the user's right side as LEFT_*. So we look up LEFT_*
        # landmarks internally but display them as "RIGHT_*".
        MP_SIDE = "LEFT"          # MediaPipe landmark key (in mirrored frame)
        DISPLAY_SIDE = "RIGHT"    # What we show to the user

        # Still gate on visibility — if the forearm isn't in frame, draw nothing.
        VIS_THRESHOLD = 0.6
        tracked_visible = all(
            pose_lm[POSE_IDX[f"{MP_SIDE}_{joint}"]].visibility >= VIS_THRESHOLD
            for joint in ("ELBOW", "WRIST")
        )

        if tracked_visible:
            # Draw a labelled dot at each tracked joint
            for joint in ("ELBOW", "WRIST"):
                idx = POSE_IDX[f"{MP_SIDE}_{joint}"]
                lx = int(pose_lm[idx].x * w)  # landmarks are normalised 0-1, scale to pixels
                ly = int(pose_lm[idx].y * h)
                cv2.circle(frame, (lx, ly), 8, (0, 255, 128), -1)
                cv2.putText(frame, f"{DISPLAY_SIDE}_{joint}", (lx + 10, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 128), 1)

            # Connect elbow -> wrist as a line (the forearm)
            pts = []
            for joint in ("ELBOW", "WRIST"):
                idx = POSE_IDX[f"{MP_SIDE}_{joint}"]
                pts.append((int(pose_lm[idx].x * w), int(pose_lm[idx].y * h)))
            cv2.polylines(frame, [np.array(pts)], False, (0, 200, 100), 2)

            '''
            FOR ANGLE CALCULATION OF FOREARM
            '''

            # Unpack the 2 pixel coords for angle calculation
            elbow_px, wrist_px = pts

            # Forearm elevation — how far the forearm is tilted above horizontal
            # Uses elbow->wrist vector vs the table plane (horizontal in image space)
            raw_elev = elevation_angle(elbow_px, wrist_px)
            elev_smoother.add(current_ms, raw_elev)
            frame_forearm_elev = elev_smoother.value()

            # Forearm yaw — horizontal sweep of the elbow->wrist vector. Uses raw
            # normalised landmark coords because MediaPipe .z (depth relative to
            # hip midpoint, negative = closer to camera) is only meaningful in
            # the normalised space. -dz in atan2 makes 0° = arm pointing at camera.
            elbow_lm = pose_lm[POSE_IDX[f"{MP_SIDE}_ELBOW"]]
            wrist_lm = pose_lm[POSE_IDX[f"{MP_SIDE}_WRIST"]]
            dx = wrist_lm.x - elbow_lm.x
            dz = wrist_lm.z - elbow_lm.z
            raw_yaw = np.degrees(np.arctan2(dx, -dz))
            # Offset so 0° = arm horizontal to camera (not pointing at it).
            raw_yaw = ((raw_yaw - 90.0 + 180.0) % 360.0) - 140.0
            raw_yaw = max(-YAW_LIMIT_DEG, min(YAW_LIMIT_DEG, raw_yaw))

            yaw_smoother.add(current_ms, raw_yaw)
            frame_forearm_yaw = yaw_smoother.value()

            # Display smoothed values near the elbow joint so the preview
            # matches what gets pushed to the server.
            ex, ey = elbow_px
            cv2.putText(frame, f"Elev: {frame_forearm_elev:.1f}deg",
                        (ex + 10, ey + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            cv2.putText(frame, f"Yaw: {frame_forearm_yaw:.1f}deg",
                        (ex + 10, ey + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

            # Save wrist position in normalised coords for hand matching below.
            # Key by the MediaPipe side so the elbow lookup in the hand section
            # (which uses POSE_IDX directly) stays consistent.
            wrist_idx = POSE_IDX[f"{MP_SIDE}_WRIST"]
            pose_wrist_positions[MP_SIDE] = (pose_lm[wrist_idx].x, pose_lm[wrist_idx].y)

    # -----------------------------------------------------------------------------
    # HAND RENDERING
    # -----------------------------------------------------------------------------
    if hands_results.hand_landmarks:
        # Zip landmarks with handedness so we know which hand is which
        for hand_lm, handedness in zip(hands_results.hand_landmarks,
                                       hands_results.handedness):
            # MediaPipe labels handedness relative to the image it sees. Because
            # we mirrored the frame with cv2.flip before inference, its "Left"
            # is actually the user's right hand, and vice versa — so swap it.
            raw_label = handedness[0].category_name  # "Left" or "Right"
            label = "Right" if raw_label == "Left" else "Left"

            raw_pct = hand_openness(hand_lm)
            hand_smoother.add(current_ms, raw_pct)
            smoothed_pct = hand_smoother.value()
            if smoothed_pct is None:
                smoothed_pct = raw_pct

            # Two-sided hysteresis on the smoothed pct: only flip the boolean
            # when the value clearly crosses one of the thresholds.
            if _hand_is_open_state:
                if smoothed_pct <= OPEN_EXIT:
                    _hand_is_open_state = False
            else:
                if smoothed_pct >= OPEN_ENTER:
                    _hand_is_open_state = True

            is_open = _hand_is_open_state
            state = "Open" if is_open else "Closed"
            color = (0, 255, 0) if is_open else (0, 0, 255)

            frame_hand_side = label
            frame_hand_pct  = smoothed_pct
            frame_hand_open = is_open

            # Draw the 21-landmark hand skeleton
            draw_hand_skeleton(frame, hand_lm, color)

            # Anchor the label to the wrist landmark position
            wrist = hand_lm[HAND_WRIST]
            wx, wy = int(wrist.x * w), int(wrist.y * h)

            # Show hand side, binary state, and smoothed openness percentage
            cv2.putText(frame, f"{label}: {state} {smoothed_pct * 100:.2f}%",
                        (wx, wy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Wrist angle relative to horizontal: elevation of the hand vector
            # (wrist -> middle-finger MCP) in image space. 0° = flat, ±90° = vertical.
            mid_mcp = hand_lm[MIDDLE_FINGER_MCP]
            mid_mcp_px = (int(mid_mcp.x * w), int(mid_mcp.y * h))
            wrist_px = (wx, wy)

            raw_wrist_bend = elevation_angle(wrist_px, mid_mcp_px)
            wrist_smoother.add(current_ms, raw_wrist_bend)
            frame_wrist_bend = wrist_smoother.value()

            cv2.putText(frame, f"Wrist angle: {frame_wrist_bend:.1f}deg",
                        (wx, wy + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

    # Push this frame's measurements out to server-control (if enabled).
    if pusher is not None:
        frame_b64 = None
        if current_ms - _last_frame_send_ms >= (1000 // DASHBOARD_FPS):
            ok, jpeg_buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                frame_b64 = base64.b64encode(jpeg_buf).decode('ascii')
                _last_frame_send_ms = current_ms

        payload = build_frame_payload(
            ts_ms=current_ms,
            forearm_elev=frame_forearm_elev,
            forearm_yaw=frame_forearm_yaw,
            wrist_bend=frame_wrist_bend,
            hand_side=frame_hand_side,
            hand_pct=frame_hand_pct,
            hand_is_open=frame_hand_open,
            frame_b64=frame_b64,
        )
        pusher.push(payload)

    cv2.imshow("Pose + Hands Tracking", frame)

    # q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release all resources cleanly
cap.release()
pose_landmarker.close()
hand_landmarker.close()
cv2.destroyAllWindows()
if pusher is not None:
    pusher.stop()
