#!/usr/bin/env python3
"""
WebSocket servo controller for the surgical training arm.
Receives pre-computed servo angles from the relay server and drives the servos.

Run on Raspberry Pi:
  sudo pigpiod
  python3 ws_servo_controller.py --url ws://RELAY_IP:8000/ws/stream

The vision app sends nested JSON each frame:
  {"forearm": {"yaw_deg": ..., "elevation_deg": ...},
   "wrist": {"bend_deg": ...},
   "hand": {"is_open": true}}

This script maps CV angles to servo angles and writes them via pigpio.
"""

import argparse
import asyncio
import json
import signal
import sys
import time

import pigpio
import websockets

# ─── GPIO Pin Assignment (from hardware design report) ───────────────────────
BASE_PIN = 12      # GPIO 12, Physical Pin 32 — YM2765 base yaw
SHOULDER_PIN = 13  # GPIO 13, Physical Pin 33 — YM2763 shoulder pitch
NEEDLE_PIN = 18    # GPIO 18, Physical Pin 12 — YM2758 needle angle

# ─── Servo Pulse Width Ranges (microseconds) ─────────────────────────────────
BASE_PWM_MIN = 640
BASE_PWM_MAX = 2000

SHOULDER_PWM_MIN = 640
SHOULDER_PWM_MAX = 2000

NEEDLE_PWM_MIN = 544
NEEDLE_PWM_MAX = 2400

# ─── Angle Limits (degrees, from hardware design report) ─────────────────────
BASE_ANGLE_MIN = -80.0
BASE_ANGLE_MAX = 80.0

SHOULDER_ANGLE_MIN = -60.0
SHOULDER_ANGLE_MAX = 0.0
SHOULDER_PHYSICAL_MIN = -60.0  # full mechanical range min
SHOULDER_PHYSICAL_MAX = 0.0    # full mechanical range max

NEEDLE_ANGLE_MIN = 0.0
NEEDLE_ANGLE_MAX = 30.0
NEEDLE_PHYSICAL_MIN = 0.0     # full mechanical range min
NEEDLE_PHYSICAL_MAX = 60.0    # full mechanical range max

# ─── CV Input Ranges (from computer vision coordinate system) ────────────────
CV_BASE_MIN = 0.0       # CV yaw at servo -80°
CV_BASE_MAX = -80.0     # CV yaw at servo +80°

CV_SHOULDER_MIN = 0.0   # CV elevation at servo -60°
CV_SHOULDER_MAX = 90.0  # CV elevation at servo 0°

CV_NEEDLE_MIN = 90.0    # CV wrist bend at servo 0°
CV_NEEDLE_MAX = -90.0   # CV wrist bend at servo 30°

# ─── Velocity Ramping ────────────────────────────────────────────────────────
MAX_VELOCITY = 120.0   # degrees/sec — max rate servos ramp toward target
RAMP_HZ = 50           # servo update loop frequency (50Hz = 20ms per tick)

# ─── Zone Classification Thresholds ──────────────────────────────────────────
# Percentage of range from the limit at which we enter amber/red zones
AMBER_THRESHOLD = 0.15   # within 15% of limit
RED_THRESHOLD = 0.05     # within 5% of limit


def angle_to_pwm(angle, angle_min, angle_max, pwm_min, pwm_max):
    """Convert an angle to a PWM pulse width in microseconds."""
    ratio = (angle - angle_min) / (angle_max - angle_min)
    return int(pwm_min + ratio * (pwm_max - pwm_min))


def clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))


def map_range(value, in_min, in_max, out_min, out_max):
    """Linearly map value from [in_min, in_max] to [out_min, out_max]."""
    ratio = (value - in_min) / (in_max - in_min)
    return out_min + ratio * (out_max - out_min)


def classify_zone(angle, angle_min, angle_max):
    """Classify angle into green/amber/red zone based on proximity to limits."""
    total_range = angle_max - angle_min
    dist_to_min = abs(angle - angle_min) / total_range
    dist_to_max = abs(angle - angle_max) / total_range
    dist_to_edge = min(dist_to_min, dist_to_max)

    if dist_to_edge <= RED_THRESHOLD:
        return "red"
    elif dist_to_edge <= AMBER_THRESHOLD:
        return "amber"
    return "green"


class ServoController:
    def __init__(self):
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError(
                "Cannot connect to pigpio daemon. Run: sudo pigpiod"
            )

        # Current angles (start at safe defaults)
        self.yaw = 0.0
        self.pitch = 0.0
        self.needle = 0.0

        # Target angles (set by CV messages, ramped toward by update loop)
        self.target_yaw = 0.0
        self.target_pitch = 0.0
        self.target_needle = 0.0

        self.clutch = False

        # Stats
        self.last_msg_time = 0.0
        self.msg_count = 0

        # Move to home position
        self._write_servos()
        print(f"[SERVO] Initialized — Home position (0°, 0°, 0°)")

    def set_target(self, cv_yaw, cv_pitch, cv_needle, clutch):
        """Set target servo positions from CV angles (does not write PWM).

        Any argument that is None is skipped — the servo holds its previous
        target.  This prevents slamming to limits when a joint is not visible.
        """
        self.clutch = clutch

        if clutch:
            # Hold current position — don't update targets
            return

        if cv_yaw is not None:
            self.target_yaw = clamp(
                map_range(cv_yaw, CV_BASE_MIN, CV_BASE_MAX,
                          BASE_ANGLE_MIN, BASE_ANGLE_MAX),
                BASE_ANGLE_MIN, BASE_ANGLE_MAX)
        if cv_pitch is not None:
            self.target_pitch = clamp(
                map_range(cv_pitch, CV_SHOULDER_MIN, CV_SHOULDER_MAX,
                          SHOULDER_ANGLE_MIN, SHOULDER_ANGLE_MAX),
                SHOULDER_ANGLE_MIN, SHOULDER_ANGLE_MAX)
        if cv_needle is not None:
            self.target_needle = clamp(
                map_range(cv_needle, CV_NEEDLE_MIN, CV_NEEDLE_MAX,
                          NEEDLE_ANGLE_MIN, NEEDLE_ANGLE_MAX),
                NEEDLE_ANGLE_MIN, NEEDLE_ANGLE_MAX)

        self.last_msg_time = time.monotonic()
        self.msg_count += 1

    def ramp_step(self, dt):
        """Move current angles toward targets at MAX_VELOCITY. Write PWM if changed."""
        if self.clutch:
            return

        moved = False
        for attr, target_attr, lo, hi in [
            ("yaw", "target_yaw", BASE_ANGLE_MIN, BASE_ANGLE_MAX),
            ("pitch", "target_pitch", SHOULDER_ANGLE_MIN, SHOULDER_ANGLE_MAX),
            ("needle", "target_needle", NEEDLE_ANGLE_MIN, NEEDLE_ANGLE_MAX),
        ]:
            current = getattr(self, attr)
            target = getattr(self, target_attr)
            diff = target - current
            max_step = MAX_VELOCITY * dt
            if abs(diff) > max_step:
                new_val = current + max_step * (1 if diff > 0 else -1)
            else:
                new_val = target
            new_val = clamp(new_val, lo, hi)
            if new_val != current:
                setattr(self, attr, new_val)
                moved = True

        if moved:
            self._write_servos()

    def _write_servos(self):
        """Write current angles to servo PWM signals."""
        self.pi.set_servo_pulsewidth(
            BASE_PIN,
            angle_to_pwm(self.yaw, BASE_ANGLE_MIN, BASE_ANGLE_MAX,
                          BASE_PWM_MIN, BASE_PWM_MAX)
        )
        self.pi.set_servo_pulsewidth(
            SHOULDER_PIN,
            angle_to_pwm(self.pitch, SHOULDER_PHYSICAL_MIN, SHOULDER_PHYSICAL_MAX,
                          SHOULDER_PWM_MIN, SHOULDER_PWM_MAX)
        )
        self.pi.set_servo_pulsewidth(
            NEEDLE_PIN,
            angle_to_pwm(self.needle, NEEDLE_PHYSICAL_MIN, NEEDLE_PHYSICAL_MAX,
                          NEEDLE_PWM_MIN, NEEDLE_PWM_MAX)
        )

    def get_status(self):
        """Return current state with zone classification."""
        return {
            "yaw": round(self.yaw, 1),
            "pitch": round(self.pitch, 1),
            "angle": round(self.needle, 1),
            "clutch": self.clutch,
            "zones": {
                "yaw": classify_zone(self.yaw, BASE_ANGLE_MIN, BASE_ANGLE_MAX),
                "pitch": classify_zone(self.pitch, SHOULDER_ANGLE_MIN, SHOULDER_ANGLE_MAX),
                "angle": classify_zone(self.needle, NEEDLE_ANGLE_MIN, NEEDLE_ANGLE_MAX),
            },
            "msg_count": self.msg_count,
        }

    def shutdown(self):
        """Disable all servo signals and disconnect."""
        print("[SERVO] Shutting down — disabling servo signals")
        self.pi.set_servo_pulsewidth(BASE_PIN, 0)
        self.pi.set_servo_pulsewidth(SHOULDER_PIN, 0)
        self.pi.set_servo_pulsewidth(NEEDLE_PIN, 0)
        self.pi.stop()


async def run(url, status_interval=0.1):
    """Connect to relay server and process incoming angle commands."""
    controller = ServoController()

    # Graceful shutdown on Ctrl+C
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.ensure_future(shutdown(controller)))

    while True:
        try:
            print(f"[WS] Connecting to {url} ...")
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                print(f"[WS] Connected")

                # Task 1: Receive WS messages, store latest target (fast, non-blocking)
                async def receive_loop():
                    while True:
                        message = await ws.recv()
                        # Drain buffered messages — keep only the latest
                        while True:
                            try:
                                message = await asyncio.wait_for(ws.recv(), timeout=0)
                            except (asyncio.TimeoutError, asyncio.CancelledError):
                                break
                        try:
                            data = json.loads(message)
                        except json.JSONDecodeError:
                            continue

                        forearm = data.get("forearm", {})
                        wrist = data.get("wrist", {})
                        hand = data.get("hand", {})

                        # Only extract angles when the joint is visible.
                        # None → set_target holds the previous target.
                        cv_yaw = forearm.get("yaw_deg") if forearm.get("visible") else None
                        cv_pitch = forearm.get("elevation_deg") if forearm.get("visible") else None
                        cv_needle = wrist.get("bend_deg") if wrist.get("visible") else None
                        clutch = not hand.get("is_open", True)

                        controller.set_target(cv_yaw, cv_pitch, cv_needle, clutch)

                        # Print status every 100 messages
                        if controller.msg_count % 100 == 0:
                            status = controller.get_status()
                            zones = status["zones"]
                            print(
                                f"[SERVO] #{controller.msg_count:>6d}  "
                                f"yaw={controller.yaw:+6.1f}° [{zones['yaw']:>5s}]  "
                                f"pitch={controller.pitch:+6.1f}° [{zones['pitch']:>5s}]  "
                                f"needle={controller.needle:+5.1f}° [{zones['angle']:>5s}]  "
                                f"clutch={'ON' if controller.clutch else 'off'}"
                            )

                # Task 2: 50Hz ramp loop, moves current toward target smoothly
                async def ramp_loop():
                    last = time.monotonic()
                    while True:
                        now = time.monotonic()
                        controller.ramp_step(now - last)
                        last = now
                        await asyncio.sleep(1.0 / RAMP_HZ)

                # Task 3: Send status back to the relay periodically
                async def send_status():
                    while True:
                        try:
                            status = controller.get_status()
                            await ws.send(json.dumps(status))
                        except Exception:
                            break
                        await asyncio.sleep(status_interval)

                try:
                    await asyncio.gather(
                        receive_loop(),
                        ramp_loop(),
                        send_status(),
                    )
                except websockets.ConnectionClosed:
                    raise  # let outer handler reconnect

        except (websockets.ConnectionClosed, ConnectionRefusedError, OSError) as e:
            print(f"[WS] Connection lost: {e}. Reconnecting in 2s...")
            await asyncio.sleep(2)
        except asyncio.CancelledError:
            break

    controller.shutdown()


async def shutdown(controller):
    """Graceful shutdown handler."""
    controller.shutdown()
    for task in asyncio.all_tasks():
        task.cancel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebSocket servo controller")
    parser.add_argument(
        "--url", type=str, default="ws://127.0.0.1:8000/ws/stream",
        help="WebSocket URL of the relay server (e.g. ws://localhost:8000/ws/stream)"
    )
    parser.add_argument(
        "--status-interval", type=float, default=0.1,
        help="How often to send status back (seconds, default 0.1)"
    )
    args = parser.parse_args()

    asyncio.run(run(args.url, args.status_interval))