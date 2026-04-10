# Setup & Run Guide

End-to-end instructions to bring up the remote surgical training arm:
laptop webcam → CV client → FastAPI relay → Raspberry Pi → servos.

## Prerequisites

- **Laptop** with a webcam, Python 3.10+, and network access to the Pi.
- **Raspberry Pi 4B** with `pigpio` installed, wired to the three servos per
  the hardware design report, and on the same LAN as the laptop.
- Both machines on the **same Wi-Fi/LAN** (not a VPN, not a guest network).

## One-time install

### Laptop

```powershell
cd C:\Users\aarav\Desktop\myCode\hsil-health-hackathon-submission\server-control
pip install -r requirements.txt

cd ..\computer-vision
pip install -r requirements.txt
```

### Raspberry Pi

```bash
sudo apt install pigpio python3-pigpio
pip3 install websockets
```

The `ws_servo_controller.py` and other Robotics scripts should live in the
Pi's home directory (`~`).

## Find the laptop's LAN IP

On the laptop:

```powershell
ipconfig
```

Look at the active adapter (Wi-Fi or Ethernet) and copy the `IPv4 Address`
line — e.g. `192.168.1.42`. Every `<LAPTOP_IP>` below means this number.

Quick reachability check from the Pi:

```bash
ping <LAPTOP_IP>
```

If ping fails, fix the network before going further.

## One-time: allow inbound port 8000 through Windows Firewall

In an **admin** PowerShell on the laptop, run once:

```powershell
New-NetFirewallRule -DisplayName "uvicorn 8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```

## Run the system

Open three separate terminals. Start them in order. Each stays running
until you Ctrl+C it.

### Terminal 1 — Relay server (laptop)

```powershell
cd C:\Users\aarav\Desktop\myCode\hsil-health-hackathon-submission\server-control
uvicorn main:app --host 0.0.0.0 --port 8000
```

Wait for `Uvicorn running on http://0.0.0.0:8000`. Leave running.

### Terminal 2 — CV client (laptop)

```powershell
cd C:\Users\aarav\Desktop\myCode\hsil-health-hackathon-submission\computer-vision
python app.py
```

An OpenCV preview window opens. Terminal 1 should log
`ingest producer connected`. Leave running.

If the wrong camera opens:

```powershell
python app.py --device 1
```

### (Optional) Sanity check from the laptop

```powershell
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/state
```

`/health` should show `"ingest_active": true`. `/state` should contain
`forearm.yaw_deg`, `forearm.elevation_deg`, `wrist.bend_deg` once you stand
in front of the camera.

### Terminal 3 — Servo controller (Raspberry Pi)

```bash
sudo pigpiod
cd ~
python3 ws_servo_controller.py --url ws://<LAPTOP_IP>:8000/ws/stream
```

Replace `<LAPTOP_IP>` with the real IP — e.g.
`ws://192.168.1.42:8000/ws/stream`.

You should see:

```
[SERVO] Initialized — Home position (0°, 0°, 0°)
[WS] Connecting to ws://192.168.1.42:8000/ws/stream ...
[WS] Connected
```

Terminal 1 will log `consumer connected (total=1)`. Move your forearm in
front of the camera — base, shoulder, and needle servos should track.
Close your hand into a fist to engage the clutch (servos freeze in place).

## Shutdown order

Ctrl+C Terminal 3 first (servos go limp cleanly), then Terminal 2, then
Terminal 1.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Name or service not known` on the Pi | You left `<LAPTOP_IP>` as literal text. Replace with the actual IP (digits and dots only). |
| `Connection refused` on the Pi | Windows Firewall blocking port 8000 — run the `New-NetFirewallRule` command above. Also confirm Terminal 1 bound to `0.0.0.0`, not `127.0.0.1`. |
| `ping` from Pi fails | Machines aren't on the same network. Check Wi-Fi SSID on both, disable VPN. |
| 404 on `/state` or `/health` | Uvicorn was launched from the wrong directory. Must be run from inside `server-control/` so `main:app` resolves. |
| Terminal 1 shows `rejected second ingest client` | The CV client is already connected from a previous run. Kill the old `python app.py` before starting a new one. |
| OpenCV window blank or wrong feed | Try `python app.py --device 1` (or 2, 3...). |
| Servos twitch to an extreme and stay there | CV→servo range constants in `ws_servo_controller.py` lines 57-65 may need a sign flip for your mechanical setup. |

## Endpoints reference

- `ws://<host>:8000/ws/ingest` — CV producer (single client only)
- `ws://<host>:8000/ws/stream` — fan-out to consumers (servo controller, dashboards)
- `GET http://<host>:8000/health` — liveness + connection counts
- `GET http://<host>:8000/state` — latest frame payload as JSON
- `http://<host>:8000/` — minimal debug viewer in the browser
