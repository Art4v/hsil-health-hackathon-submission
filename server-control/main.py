"""
FastAPI bridge between the MediaPipe CV process and downstream consumers.

- /ws/ingest : single producer (the CV app) pushes one JSON payload per frame.
- /ws/stream : any number of consumers receive every payload as it arrives.
- /state     : REST snapshot of the most recent payload.
- /health    : liveness + basic stats.
"""

import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from starlette.websockets import WebSocketState

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("server-control")

app = FastAPI(title="server-control")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
DASHBOARD_PATH = FRONTEND_DIR / "dashboard.html"

# Latest frame payload pushed by the CV producer. None when no producer connected.
latest_state: dict | None = None

# Set of active consumer websockets receiving the broadcast.
consumers: set[WebSocket] = set()

# Enforce a single producer on /ws/ingest.
ingest_active: bool = False


INDEX_HTML = """<!doctype html>
<title>server-control viewer</title>
<h1>server-control viewer</h1>
<p>Status: <span id="status">connecting...</span></p>
<p>Timestamp (ms): <span id="ts">-</span></p>
<h2>Forearm</h2>
<p>Visible: <span id="forearm_vis">-</span></p>
<p>Elevation (deg): <span id="forearm_elev">-</span></p>
<p>Yaw (deg): <span id="forearm_yaw">-</span></p>
<h2>Wrist</h2>
<p>Visible: <span id="wrist_vis">-</span></p>
<p>Bend (deg): <span id="wrist_bend">-</span></p>
<h2>Hand</h2>
<p>Visible: <span id="hand_vis">-</span></p>
<p>Side: <span id="hand_side">-</span></p>
<p>Openness (%): <span id="hand_pct">-</span></p>
<p>Open: <span id="hand_open">-</span></p>
<h2>Raw</h2>
<pre id="raw">-</pre>
<script>
const $ = id => document.getElementById(id);
function set(id, v) { $(id).textContent = v; }
function fmt(n) { return (typeof n === "number") ? n.toFixed(1) : "-"; }

function connect() {
  const ws = new WebSocket("ws://" + location.host + "/ws/stream");
  ws.onopen    = () => set("status", "connected");
  ws.onclose   = () => { set("status", "disconnected, retrying..."); setTimeout(connect, 1000); };
  ws.onerror   = () => set("status", "error");
  ws.onmessage = (ev) => {
    const d = JSON.parse(ev.data);
    set("ts", d.ts_ms);
    set("forearm_vis",  d.forearm.visible);
    set("forearm_elev", fmt(d.forearm.elevation_deg));
    set("forearm_yaw",  fmt(d.forearm.yaw_deg));
    set("wrist_vis",    d.wrist.visible);
    set("wrist_bend",   fmt(d.wrist.bend_deg));
    set("hand_vis",     d.hand.visible);
    set("hand_side",    d.hand.side ?? "-");
    set("hand_pct",     (typeof d.hand.openness_pct === "number") ? (d.hand.openness_pct * 100).toFixed(1) : "-");
    set("hand_open",    (typeof d.hand.is_open === "boolean") ? d.hand.is_open : "-");
    set("raw",          JSON.stringify(d, null, 2));
  };
}
connect();
</script>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return INDEX_HTML


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_PATH.read_text(encoding="utf-8")


@app.get("/health")
async def health():
    return {
        "ok": True,
        "ingest_active": ingest_active,
        "consumers": len(consumers),
    }


@app.get("/state")
async def state():
    if latest_state is None:
        return {"status": "no signal"}
    return latest_state


async def _broadcast(payload: dict) -> None:
    """Fan out a payload to every consumer; drop any that fail."""
    if not consumers:
        return
    dead: list[WebSocket] = []
    # Snapshot the set so mutation during iteration is safe.
    for ws in list(consumers):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        consumers.discard(ws)


@app.websocket("/ws/ingest")
async def ws_ingest(ws: WebSocket):
    global ingest_active, latest_state

    await ws.accept()

    if ingest_active:
        # Single-producer policy: reject anyone showing up second.
        await ws.close(code=1008, reason="ingest already active")
        log.warning("rejected second ingest client")
        return

    ingest_active = True
    log.info("ingest producer connected")

    try:
        while True:
            payload = await ws.receive_json()
            if not isinstance(payload, dict) or "ts_ms" not in payload:
                # Malformed frame — ignore rather than tear down the stream.
                continue
            latest_state = payload
            await _broadcast(payload)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.exception("ingest error: %s", e)
    finally:
        ingest_active = False
        latest_state = None
        log.info("ingest producer disconnected")


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    consumers.add(ws)
    log.info("consumer connected (total=%d)", len(consumers))

    try:
        # Prime the new consumer with the current snapshot, if any.
        if latest_state is not None:
            await ws.send_json(latest_state)

        # Stay connected until the client goes away. We don't expect inbound
        # messages, but receiving gives us a clean disconnect signal.
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.exception("stream error: %s", e)
    finally:
        consumers.discard(ws)
        if ws.client_state != WebSocketState.DISCONNECTED:
            try:
                await ws.close()
            except Exception:
                pass
        log.info("consumer disconnected (total=%d)", len(consumers))
