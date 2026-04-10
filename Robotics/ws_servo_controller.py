#!/usr/bin/env python3
"""
WebSocket servo controller for the surgical training arm.
Receives pre-computed servo angles from the relay server and drives the servos.

Run on Raspberry Pi:
  sudo pigpiod
  python3 ws_servo_controller.py --url ws://RELAY_IP:8000/ws/robot

The vision app sends JSON each frame:
  {"yaw": 15.3, "pitch": -22.7, "angle": 25.0, "clutch": false}

This script writes the angles directly to the servos via pigpio.
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

SHOULDER_ANGLE_MIN = -20.0
SHOULDER_ANGLE_MAX = 0.0
SHOULDER_PHYSICAL_MIN = -60.0  # full mechanical range min
SHOULDER_PHYSICAL_MAX = 0.0    # full mechanical range max

NEEDLE_ANGLE_MIN = 0.0
NEEDLE_ANGLE_MAX = 15.0
NEEDLE_PHYSICAL_MIN = 0.0     # full mechanical range min
NEEDLE_PHYSICAL_MAX = 60.0    # full mechanical range max

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
        self.clutch = False

        # Stats
        self.last_msg_time = 0.0
        self.msg_count = 0

        # Move to home position
        self._write_servos()
        print(f"[SERVO] Initialized — Home position (0°, 0°, 0°)")

    def update(self, yaw, pitch, needle_angle, clutch):
        """Update servo positions from received angles."""
        self.clutch = clutch

        if clutch:
            # Hold current position — don't update angles
            return

        self.yaw = clamp(yaw, BASE_ANGLE_MIN, BASE_ANGLE_MAX)
        self.pitch = clamp(pitch, SHOULDER_ANGLE_MIN, SHOULDER_ANGLE_MAX)
        self.needle = clamp(needle_angle, NEEDLE_ANGLE_MIN, NEEDLE_ANGLE_MAX)

        self._write_servos()

        self.last_msg_time = time.monotonic()
        self.msg_count += 1

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

                # Task to send status back to the relay periodically
                async def send_status():
                    while True:
                        try:
                            status = controller.get_status()
                            await ws.send(json.dumps(status))
                        except Exception:
                            break
                        await asyncio.sleep(status_interval)

                status_task = asyncio.create_task(send_status())

                try:
                    async for message in ws:
                        try:
                            data = json.loads(message)
                        except json.JSONDecodeError:
                            continue

                        # Expected: {"yaw": float, "pitch": float, "angle": float, "clutch": bool}
                        yaw = data.get("yaw", 0.0)
                        pitch = data.get("pitch", 0.0)
                        needle = data.get("angle", 0.0)
                        clutch = data.get("clutch", False)

                        controller.update(yaw, pitch, needle, clutch)

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
                finally:
                    status_task.cancel()

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
        "--url", type=str, required=True,
        help="WebSocket URL of the relay server (e.g. ws://localhost:8000/ws/robot)"
    )
    parser.add_argument(
        "--status-interval", type=float, default=0.1,
        help="How often to send status back (seconds, default 0.1)"
    )
    args = parser.parse_args()

    asyncio.run(run(args.url, args.status_interval))
