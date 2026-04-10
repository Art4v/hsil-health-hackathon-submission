# Computer Vision — Setup

Real-time upper-body pose and hand tracking built on the **MediaPipe Tasks API**
(`PoseLandmarker` + `HandLandmarker`). Draws the six upper-body joints, elbow
flexion / forearm elevation angles, and a colour-coded hand skeleton with a
per-finger open/closed breakdown and wrist-bend angle.

## Requirements

- Python **3.9 – 3.12** (MediaPipe does not yet ship wheels for 3.13 on all
  platforms)
- A working webcam
- ~15 MB of disk space for the MediaPipe model files (auto-downloaded)

## Install

From the `computer-vision/` folder:

```bash
# 1. Create and activate a virtual environment
python -m venv .venv

# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (cmd / git-bash)
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

On the **first run** the script downloads two model files into
`computer-vision/models/`:

- `pose_landmarker_lite.task`
- `hand_landmarker.task`

These are cached, so subsequent runs start instantly. Press **`q`** in the
preview window to quit.

## CLI flags

| Flag | Default | Description |
| --- | --- | --- |
| `--device` | `0` | Camera index passed to `cv2.VideoCapture`. Try `1`, `2`, … if you have multiple cameras. |
| `--width` | `960` | Capture width in pixels. |
| `--height` | `540` | Capture height in pixels. |
| `--use_static_image_mode` | off | Switch landmarkers from `VIDEO` to `IMAGE` running mode (no temporal smoothing). |
| `--min_detection_confidence` | `0.5` | Minimum confidence for initial pose/hand detection. |
| `--min_tracking_confidence` | `0.5` | Minimum confidence for frame-to-frame tracking. |

Examples:

```bash
# Second webcam
python app.py --device 1

# 720p capture
python app.py --width 1280 --height 720

# Stricter detection threshold
python app.py --min_detection_confidence 0.7
```

## What you'll see

- **Green dots + labels** on both shoulders, elbows, and wrists.
- **Green polylines** connecting shoulder → elbow → wrist on each arm.
- **`Flex:` and `Elev:`** text near each elbow (elbow flexion angle and forearm
  elevation above horizontal, both in degrees).
- **Hand skeleton** drawn in:
  - **green** when the hand is Open (4–5 fingers extended)
  - **orange** when Partial (2–3 fingers extended)
  - **red** when Closed (0–1 fingers extended)
- **`Left/Right: State (n/5)`** label above each wrist.
- **`T I M R P`** per-finger breakdown below the wrist (letter = extended,
  underscore = curled).
- **`Wrist bend: Xdeg`** when a hand can be matched to a detected pose arm
  (180° = perfectly aligned with the forearm).

## Troubleshooting

**"Could not open camera device N"**
Try a different `--device` index. On Windows, close any other app that may be
holding the camera (Zoom, Teams, browser tabs, etc.).

**Model download fails (firewall / offline machine)**
Manually download these two files into `computer-vision/models/` and re-run:

- <https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task>
- <https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task>

**`ImportError: cannot import name 'vision' from 'mediapipe.tasks.python'`**
Your MediaPipe is too old. Upgrade:

```bash
pip install --upgrade "mediapipe>=0.10.14"
```

**Laggy video**
Lower the capture resolution (`--width 640 --height 360`) or raise the
detection threshold (`--min_detection_confidence 0.7`) to reduce false
positives and redraw work.
