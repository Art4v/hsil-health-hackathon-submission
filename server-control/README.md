# server-control

FastAPI bridge between the MediaPipe CV process (`../computer-vision/app.py`)
and any downstream consumers.

## Run

```
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000
```

## Endpoints

- `GET  /health`     — liveness + whether a producer is connected + consumer count
- `GET  /state`      — most recent frame payload (or `{"status": "no signal"}`)
- `WS   /ws/ingest`  — single producer pushes one JSON payload per CV frame
- `WS   /ws/stream`  — any number of consumers receive every payload as it arrives

## Frame payload

```json
{
  "ts_ms": 1733952000123,
  "forearm": { "elevation_deg": 12.4, "visible": true },
  "wrist":   { "bend_deg": 172.8, "visible": true },
  "hand":    { "side": "Right", "openness_pct": 0.87, "is_open": true, "visible": true }
}
```

Sub-objects with `visible: false` omit their numeric fields.
