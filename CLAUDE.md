
# Remote Surgical Training Arm — CV & Server

## Project Overview

Teleoperated robotic arm for remote venipuncture training. A laptop webcam tracks the operator's hand via MediaPipe; extracted angles are relayed over WebSocket to a Raspberry Pi 4B driving three Jaycar servos (2-DOF + needle angle). Built for a 24–48 hour hackathon.

## Architecture

```
Laptop webcam → app.py (MediaPipe Hands + Pose) → WebSocket → server.py (FastAPI relay) → Pi → pigpio → servos
```

**Four layers:**
1. **CV client** (`app.py`) — MediaPipe Tasks API, OpenCV, runs on laptop. Extracts forearm elevation, wrist bend, hand openness. Pushes JSON frames to the relay server.
2. **Relay server** (`server.py`) — FastAPI + WebSockets. Single-producer ingest (`/ws/ingest`), fan-out to N consumers (`/ws/stream`). Deployed on Railway.
3. **Pi controller** (not in this repo yet) — receives frames, maps angles to PWM via pigpio on GPIO 12/13/18.
4. **Hardware** — YM2765 (base yaw), YM2763 (shoulder pitch), YM2758/SG90 (needle angle). Powered by Samsung EP-TA200 5V 2A charger. 3D-printed PLA structure (6 OpenSCAD parts).

## Key Files

| File | Purpose |
|---|---|
| `app.py` | CV pipeline — pose + hand landmark detection, angle computation, WebSocket push |
| `server.py` | FastAPI relay server — ingest, broadcast, REST state, health, debug viewer |
| `hardware_design_report.docx` | Full mechanical/electrical spec, BOM, assembly sequence, safety |

## Tech Stack

- **CV:** MediaPipe Tasks API (PoseLandmarker + HandLandmarker), OpenCV, NumPy
- **Server:** FastAPI, Starlette WebSockets, uvicorn
- **Client push:** `websockets` library (async, runs on background thread)
- **Hardware:** Raspberry Pi 4B, pigpio, Jaycar YM2765/YM2763/YM2758 servos
- **Frontend (planned):** React 19 + Vite, `@mediapipe/hands` browser-side

## Running

### CV client
```bash
# Install deps
pip install mediapipe opencv-python numpy websockets

# First run auto-downloads .task model files to ./models/
python app.py                        # default: cam 0, 960×540, pushes to ws://127.0.0.1:8000/ws/ingest
python app.py --no_server            # local preview only, no WebSocket push
python app.py --device 1 --width 1280 --height 720
```
Press `q` to quit.

### Relay server
```bash
pip install fastapi uvicorn
uvicorn server:app --host 0.0.0.0 --port 8000
```
- Debug viewer: `http://localhost:8000/`
- Health: `GET /health`
- Latest frame: `GET /state`

## CV Pipeline Details

### Signals extracted per frame
- **Forearm elevation** — angle of elbow→wrist vector above horizontal (degrees). Uses pose landmarks `LEFT_ELBOW`/`LEFT_WRIST` (mirrored frame, so LEFT = user's right).
- **Wrist bend** — deviation of hand direction (wrist→middle MCP) from forearm direction (elbow→wrist). 180° = straight, 90° = fully bent. Requires both pose and hand landmarks.
- **Hand openness** — ratio of fingertip-to-wrist vs PIP-to-wrist distances, averaged over 4 fingers (thumb excluded). Linearly mapped from [0.65, 1.35] ratio range to [0%, 100%]. ≥50% = open.

### Frame payload schema
```json
{
  "ts_ms": 1234,
  "forearm": { "visible": true, "elevation_deg": 15.3 },
  "wrist":   { "visible": true, "bend_deg": 162.0 },
  "hand":    { "visible": true, "side": "Right", "openness_pct": 0.82, "is_open": true }
}
```
Fields like `elevation_deg`, `bend_deg`, `side`, `openness_pct`, `is_open` are only present when `visible: true`.

### Mirror convention
The frame is flipped (`cv2.flip(frame, 1)`) before inference so the preview mirrors the user. This means MediaPipe's "Left" labels correspond to the user's right side. The code swaps labels for display but uses the original MediaPipe indices internally.

## Server Design

- **Single-producer policy:** Only one CV client can connect to `/ws/ingest` at a time. A second connection is rejected with code 1008.
- **Fan-out broadcast:** Every frame received on ingest is forwarded to all connected `/ws/stream` consumers. Dead sockets are pruned automatically.
- **Snapshot on connect:** New consumers immediately receive the latest frame payload (if any) so they don't have to wait for the next CV frame.
- **Stateless REST:** `GET /state` returns the most recent payload or `{"status": "no signal"}`.

## Hardware Quick Reference

| Joint | Servo | GPIO | Limits |
|---|---|---|---|
| Base yaw | YM2765 (11 kg·cm) | GPIO 12 (pin 32) | ±80° |
| Shoulder pitch | YM2763 (13 kg·cm) | GPIO 13 (pin 33) | 0° to −60° |
| Needle angle | YM2758/SG90 (1.6 kg·cm) | GPIO 18 (pin 12) | 0° to 60° |

Servo power: Samsung EP-TA200 5V 2A via cut USB cable. Pi power: separate XC9122 5.1V 3A USB-C PSU. **Only GND is shared between the two supplies** (Pi pin 6). Never connect charger +5V to any GPIO pin.

Max servo velocity: 30°/sec (software-enforced).

## Outstanding Work

- Pi-side WebSocket handler + servo control code
- React dashboard (hand tracking overlay, angle gauges, video feed, alerts)
- Needle angle tracking with zone feedback
- 3D print all 6 parts, verify servo dims with calipers
- End-to-end integration test over local network
- Railway deployment for cloud relay

## Conventions

- Python 3.10+. Type hints where practical.
- MediaPipe Tasks API only (not the legacy `mp.solutions` API — dropped in newer releases).
- Model files (`.task`) are auto-downloaded to `./models/` and gitignored.
- All angles in degrees. All coordinates normalised [0, 1] unless converted to pixels for rendering.
- Frame mirror flip happens once at capture time; all downstream code operates on the mirrored frame.