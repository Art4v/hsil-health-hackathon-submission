# Library imports
import argparse
import os
import time
import urllib.request

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
    "LEFT_SHOULDER":  11,
    "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW":     13,
    "RIGHT_ELBOW":    14,
    "LEFT_WRIST":     15,
    "RIGHT_WRIST":    16,
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

# Hand openness: linear map from avg wrist-tip/wrist-pip ratio → [0, 1].
# Thumb is deliberately excluded because it moves laterally and biases the score.
OPENNESS_MIN_RATIO = 0.65   # ratio at which we call it 0%
OPENNESS_MAX_RATIO = 1.35   # ratio at which we call it 100%
OPEN_THRESHOLD     = 0.5    # ≥50% openness → "Open"

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
    Returns (pct, is_open).
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
        return 0.0, False

    avg = sum(ratios) / len(ratios)
    pct = (avg - OPENNESS_MIN_RATIO) / (OPENNESS_MAX_RATIO - OPENNESS_MIN_RATIO)
    pct = max(0.0, min(1.0, pct))
    return pct, pct >= OPEN_THRESHOLD


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
# RUN MODELS WHILE WEBCAM IS RUNNING
# -----------------------------------------------------------------------------
start_ns = time.monotonic_ns()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip so the view mirrors the user (more intuitive for self-monitoring)
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # MediaPipe Tasks API expects an mp.Image wrapping an RGB ndarray
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Run both landmarkers on the same frame
    if running_mode == VisionRunningMode.VIDEO:
        timestamp_ms = (time.monotonic_ns() - start_ns) // 1_000_000
        pose_results = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        hands_results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
    else:
        pose_results = pose_landmarker.detect(mp_image)
        hands_results = hand_landmarker.detect(mp_image)

    # Store pose wrist positions so the hand section can look them up:
    # { "LEFT": (x, y), "RIGHT": (x, y) } in normalised coords
    pose_wrist_positions = {}
    pose_lm = None  # populated below if a pose was detected

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

        # Still gate on visibility — if the tracked arm isn't in frame, draw nothing.
        VIS_THRESHOLD = 0.6
        tracked_visible = all(
            pose_lm[POSE_IDX[f"{MP_SIDE}_{joint}"]].visibility >= VIS_THRESHOLD
            for joint in ("SHOULDER", "ELBOW", "WRIST")
        )

        if tracked_visible:
            # Draw a labelled dot at each tracked joint
            for joint in ("SHOULDER", "ELBOW", "WRIST"):
                idx = POSE_IDX[f"{MP_SIDE}_{joint}"]
                lx = int(pose_lm[idx].x * w)  # landmarks are normalised 0-1, scale to pixels
                ly = int(pose_lm[idx].y * h)
                cv2.circle(frame, (lx, ly), 8, (0, 255, 128), -1)
                cv2.putText(frame, f"{DISPLAY_SIDE}_{joint}", (lx + 10, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 128), 1)

            # Connect shoulder -> elbow -> wrist as a polyline
            pts = []
            for joint in ("SHOULDER", "ELBOW", "WRIST"):
                idx = POSE_IDX[f"{MP_SIDE}_{joint}"]
                pts.append((int(pose_lm[idx].x * w), int(pose_lm[idx].y * h)))
            cv2.polylines(frame, [np.array(pts)], False, (0, 200, 100), 2)

            '''
            FOR ANGLE CALCULATION OF ARM
            '''

            # Unpack the 3 pixel coords for angle calculations
            shoulder_px, elbow_px, wrist_px = pts

            # Elbow flexion angle — angle at elbow between upper arm and forearm
            # 180° = fully straight, smaller = more bent
            elbow_angle = angle_between(shoulder_px, elbow_px, wrist_px)

            # Forearm elevation — how far the forearm is tilted above horizontal
            # Uses elbow->wrist vector vs the table plane (horizontal in image space)
            forearm_elev = elevation_angle(elbow_px, wrist_px)

            # Display both angles near the elbow joint
            ex, ey = elbow_px
            cv2.putText(frame, f"Flex: {elbow_angle:.1f}deg",
                        (ex + 10, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            cv2.putText(frame, f"Elev: {forearm_elev:.1f}deg",
                        (ex + 10, ey + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

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

            pct, is_open = hand_openness(hand_lm)
            state = "Open" if is_open else "Closed"
            color = (0, 255, 0) if is_open else (0, 0, 255)

            # Draw the 21-landmark hand skeleton
            draw_hand_skeleton(frame, hand_lm, color)

            # Anchor the label to the wrist landmark position
            wrist = hand_lm[HAND_WRIST]
            wx, wy = int(wrist.x * w), int(wrist.y * h)

            # Show hand side, binary state, and continuous openness percentage
            cv2.putText(frame, f"{label}: {state} {pct * 100:.2f}%",
                        (wx, wy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Compute wrist bend angle using elbow (from pose) -> wrist -> middle MCP.
            # First, match this hand to the correct pose arm by finding the closest
            # pose wrist to the hand's wrist landmark (in normalised coords)
            if pose_wrist_positions and pose_lm is not None:
                hand_wrist_norm = np.array([wrist.x, wrist.y])
                closest_side = min(
                    pose_wrist_positions,
                    key=lambda s: np.linalg.norm(
                        hand_wrist_norm - np.array(pose_wrist_positions[s])
                    )
                )

                elbow_idx = POSE_IDX[f"{closest_side}_ELBOW"]

                # Convert pose elbow to pixels
                elbow_px = (
                    int(pose_lm[elbow_idx].x * w),
                    int(pose_lm[elbow_idx].y * h)
                )
                wrist_px = (wx, wy)

                # Middle finger MCP = base knuckle of middle finger
                # Acts as a proxy for the direction of the hand beyond the wrist
                mid_mcp = hand_lm[MIDDLE_FINGER_MCP]
                mid_mcp_px = (int(mid_mcp.x * w), int(mid_mcp.y * h))

                # Angle at wrist: how much the hand deviates from the forearm direction
                # 180° = hand perfectly aligned with forearm, smaller = wrist is bent
                wrist_bend = angle_between(elbow_px, wrist_px, mid_mcp_px)

                cv2.putText(frame, f"Wrist bend: {wrist_bend:.1f}deg",
                            (wx, wy + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

    cv2.imshow("Pose + Hands Tracking", frame)

    # q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release all resources cleanly
cap.release()
pose_landmarker.close()
hand_landmarker.close()
cv2.destroyAllWindows()
