#!/usr/bin/env python3
"""
Centre all three servos to their home positions.
Run on Raspberry Pi: sudo pigpiod && python3 centre_servos.py
"""

import time
import pigpio

# GPIO pins (from hardware design report)
BASE_PIN = 12       # GPIO 12 — YM2765 base yaw
SHOULDER_PIN = 13   # GPIO 13 — YM2763 shoulder pitch
NEEDLE_PIN = 18     # GPIO 18 — YM2758 needle angle

# Home = centre of each servo's range
# Base: ±80° → centre is 0° → neutral 1300 µs
# Shoulder: 0° to -60° → home is 0° (horizontal) → PWM at max end of range
# Needle: 0° to 60° → home is 0° (retracted) → PWM at min end of range
BASE_HOME_PWM = 1300      # 0° centre
SHOULDER_HOME_PWM = 2000  # 0° horizontal (top of range: -60°=640, 0°=2000)
NEEDLE_HOME_PWM = 544     # 0° retracted (500 µs causes jitter on SG90)

pi = pigpio.pi()
if not pi.connected:
    raise RuntimeError("Cannot connect to pigpio daemon. Run: sudo pigpiod")

print("Centring servos...")
pi.set_servo_pulsewidth(BASE_PIN, BASE_HOME_PWM)
pi.set_servo_pulsewidth(SHOULDER_PIN, SHOULDER_HOME_PWM)
pi.set_servo_pulsewidth(NEEDLE_PIN, NEEDLE_HOME_PWM)

print(f"  Base     → {BASE_HOME_PWM} µs  (0°)")
print(f"  Shoulder → {SHOULDER_HOME_PWM} µs  (0° horizontal)")
print(f"  Needle   → {NEEDLE_HOME_PWM} µs  (0° retracted)")

# Hold for 2 seconds so servos reach position, then release
time.sleep(2)
pi.set_servo_pulsewidth(BASE_PIN, 0)
pi.set_servo_pulsewidth(SHOULDER_PIN, 0)
pi.set_servo_pulsewidth(NEEDLE_PIN, 0)
pi.stop()
print("Done — signals released.")
