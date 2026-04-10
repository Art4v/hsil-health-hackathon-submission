"""
FastAPI WebSocket relay server for the surgical training arm.

Two endpoints:
  /ws/control  — the vision/hand-tracking client connects here (sends servo angles)
  /ws/robot    — the Raspberry Pi connects here (receives servo angles, sends status)

Messages from control clients are forwarded to all connected robots, and vice versa.

Run locally:
  uvicorn server:app --host 0.0.0.0 --port 8000

Deploy on Railway:
  Railway auto-detects the Procfile.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import asyncio
import json
import time

app = FastAPI(title="Surgical Arm Relay")

# Connected clients
control_clients: list[WebSocket] = []
robot_clients: list[WebSocket] = []


@app.get("/")
async def root():
    return JSONResponse({
        "status": "ok",
        "controllers": len(control_clients),
        "robots": len(robot_clients),
    })


@app.get("/health")
async def health():
    return {"status": "healthy"}


async def broadcast(targets: list[WebSocket], message: str):
    """Send a message to all connected clients in a list, removing dead ones."""
    disconnected = []
    for ws in targets:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in targets:
            targets.remove(ws)


@app.websocket("/ws/control")
async def control_endpoint(websocket: WebSocket):
    """Vision/hand-tracking client connects here to send servo angle commands."""
    await websocket.accept()
    control_clients.append(websocket)
    print(f"[RELAY] Controller connected ({len(control_clients)} total)")
    try:
        while True:
            data = await websocket.receive_text()
            # Forward to all robots
            await broadcast(robot_clients, data)
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in control_clients:
            control_clients.remove(websocket)
        print(f"[RELAY] Controller disconnected ({len(control_clients)} remaining)")


@app.websocket("/ws/robot")
async def robot_endpoint(websocket: WebSocket):
    """Raspberry Pi connects here to receive servo angle commands."""
    await websocket.accept()
    robot_clients.append(websocket)
    print(f"[RELAY] Robot connected ({len(robot_clients)} total)")
    try:
        while True:
            data = await websocket.receive_text()
            # Forward robot status back to all controllers
            await broadcast(control_clients, data)
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in robot_clients:
            robot_clients.remove(websocket)
        print(f"[RELAY] Robot disconnected ({len(robot_clients)} remaining)")


# Direct connection mode: single endpoint that pairs the first two connections
# (useful for local testing without separate /control and /robot paths)
@app.websocket("/ws/direct")
async def direct_endpoint(websocket: WebSocket):
    """
    Simple pass-through for local testing.
    First client to connect is the 'sender', second is the 'receiver'.
    Messages from sender are forwarded to receiver and vice versa.
    """
    await websocket.accept()
    # Reuse the control/robot lists — first connection is control, second is robot
    if not control_clients:
        control_clients.append(websocket)
        role = "controller"
    else:
        robot_clients.append(websocket)
        role = "robot"

    print(f"[RELAY] Direct client connected as {role}")
    try:
        while True:
            data = await websocket.receive_text()
            if role == "controller":
                await broadcast(robot_clients, data)
            else:
                await broadcast(control_clients, data)
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in control_clients:
            control_clients.remove(websocket)
        if websocket in robot_clients:
            robot_clients.remove(websocket)
        print(f"[RELAY] Direct client ({role}) disconnected")
