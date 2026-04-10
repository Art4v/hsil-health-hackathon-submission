#!/usr/bin/env python3
"""
Manual servo control for the surgical training arm.
Run on Raspberry Pi over SSH. Requires pigpio daemon running (sudo pigpiod).

Controls:
  A / D  — Base yaw (left / right)
  W / S  — Shoulder pitch (up / down)
  Q / E  — Needle angle (retract / extend)
  [ / ]  — Decrease / increase velocity
  Space  — Emergency stop (hold all servos at current position)
  0      — Return all servos to home position
  Esc    — Quit
"""

import curses
import time
import pigpio

# ─── GPIO Pin Assignment (from hardware design report) ───────────────────────
BASE_PIN = 12      # GPIO 12, Physical Pin 32 — YM2765 base yaw
SHOULDER_PIN = 13  # GPIO 13, Physical Pin 33 — YM2763 shoulder pitch
NEEDLE_PIN = 18    # GPIO 18, Physical Pin 12 — YM2758 needle angle

# ─── Servo Pulse Width Ranges (microseconds) ─────────────────────────────────
# YM2765 / YM2763: ~640–2000 µs, neutral ~1300 µs
BASE_PWM_MIN = 640
BASE_PWM_MAX = 2000
BASE_PWM_NEUTRAL = 1300

SHOULDER_PWM_MIN = 640
SHOULDER_PWM_MAX = 2000
SHOULDER_PWM_NEUTRAL = 1300

# YM2758 (SG90): ~544–2400 µs (500 µs causes jitter at the range edge)
NEEDLE_PWM_MIN = 544
NEEDLE_PWM_MAX = 2400
NEEDLE_PWM_NEUTRAL = 544  # 0° = fully retracted

# ─── Angle Limits (degrees, from hardware design report) ─────────────────────
BASE_ANGLE_MIN = -80.0    # full left
BASE_ANGLE_MAX = 80.0     # full right
BASE_ANGLE_HOME = 0.0     # center

SHOULDER_ANGLE_MIN = -20.0  # max downward pitch
SHOULDER_ANGLE_MAX = 0.0    # horizontal
SHOULDER_ANGLE_HOME = 0.0   # horizontal
SHOULDER_PHYSICAL_MIN = -60.0  # full mechanical range min
SHOULDER_PHYSICAL_MAX = 0.0    # full mechanical range max

NEEDLE_ANGLE_MIN = 0.0     # retracted
NEEDLE_ANGLE_MAX = 15.0    # extended
NEEDLE_ANGLE_HOME = 0.0    # retracted
NEEDLE_PHYSICAL_MIN = 0.0     # full mechanical range min
NEEDLE_PHYSICAL_MAX = 60.0    # full mechanical range max

# ─── Velocity Settings ───────────────────────────────────────────────────────
VELOCITY_MIN = 10.0          # degrees per second
VELOCITY_MAX = 60.0         # degrees per second (safety cap from design report)
VELOCITY_DEFAULT = 30.0     # starting velocity
VELOCITY_STEP = 5.0         # increment per keypress

# ─── Loop Timing ──────────────────────────────────────────────────────────────
LOOP_HZ = 50  # control loop frequency


def angle_to_pwm(angle, angle_min, angle_max, pwm_min, pwm_max):
    """Convert an angle to a PWM pulse width in microseconds."""
    ratio = (angle - angle_min) / (angle_max - angle_min)
    return int(pwm_min + ratio * (pwm_max - pwm_min))


def clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))


def main(stdscr):
    # curses setup
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(int(1000 / LOOP_HZ))

    # Connect to pigpio daemon
    pi = pigpio.pi()
    if not pi.connected:
        stdscr.addstr(0, 0, "ERROR: Cannot connect to pigpio daemon.")
        stdscr.addstr(1, 0, "Run: sudo pigpiod")
        stdscr.refresh()
        stdscr.nodelay(False)
        stdscr.getch()
        return

    # Current servo angles
    base_angle = BASE_ANGLE_HOME
    shoulder_angle = SHOULDER_ANGLE_HOME
    needle_angle = NEEDLE_ANGLE_HOME
    velocity = VELOCITY_DEFAULT  # degrees per second

    # Movement state: tracks which directions are active
    base_dir = 0       # -1 = left (A), +1 = right (D)
    shoulder_dir = 0   # +1 = up (W), -1 = down (S)
    needle_dir = 0     # -1 = retract (Q), +1 = extend (E)
    e_stop = False

    dt = 1.0 / LOOP_HZ
    last_time = time.monotonic()

    # Set initial positions
    pi.set_servo_pulsewidth(BASE_PIN, angle_to_pwm(
        base_angle, BASE_ANGLE_MIN, BASE_ANGLE_MAX, BASE_PWM_MIN, BASE_PWM_MAX))
    pi.set_servo_pulsewidth(SHOULDER_PIN, angle_to_pwm(
        shoulder_angle, SHOULDER_PHYSICAL_MIN, SHOULDER_PHYSICAL_MAX,
        SHOULDER_PWM_MIN, SHOULDER_PWM_MAX))
    pi.set_servo_pulsewidth(NEEDLE_PIN, angle_to_pwm(
        needle_angle, NEEDLE_PHYSICAL_MIN, NEEDLE_PHYSICAL_MAX,
        NEEDLE_PWM_MIN, NEEDLE_PWM_MAX))

    try:
        while True:
            now = time.monotonic()
            dt = now - last_time
            last_time = now

            # ── Read all buffered keys ────────────────────────────────────
            # Reset directions each frame (key must be held for continuous movement)
            base_dir = 0
            shoulder_dir = 0
            needle_dir = 0

            # Drain all queued keypresses this frame
            keys_this_frame = set()
            while True:
                ch = stdscr.getch()
                if ch == -1:
                    break
                keys_this_frame.add(ch)

            for ch in keys_this_frame:
                if ch == 27:  # Esc
                    return
                elif ch == ord(' '):
                    e_stop = not e_stop
                elif ch == ord('0'):
                    # Return to home
                    base_angle = BASE_ANGLE_HOME
                    shoulder_angle = SHOULDER_ANGLE_HOME
                    needle_angle = NEEDLE_ANGLE_HOME
                    e_stop = False
                elif ch in (ord('['), ord('-')):
                    velocity = clamp(velocity - VELOCITY_STEP,
                                     VELOCITY_MIN, VELOCITY_MAX)
                elif ch in (ord(']'), ord('='), ord('+')):
                    velocity = clamp(velocity + VELOCITY_STEP,
                                     VELOCITY_MIN, VELOCITY_MAX)
                # Movement keys
                elif ch in (ord('a'), ord('A')):
                    base_dir = -1
                elif ch in (ord('d'), ord('D')):
                    base_dir = 1
                elif ch in (ord('w'), ord('W')):
                    shoulder_dir = 1
                elif ch in (ord('s'), ord('S')):
                    shoulder_dir = -1
                elif ch in (ord('q'), ord('Q')):
                    needle_dir = -1
                elif ch in (ord('e'), ord('E')):
                    needle_dir = 1

            # ── Update angles ─────────────────────────────────────────────
            if not e_stop:
                step = velocity * dt
                base_angle = clamp(base_angle + base_dir * step,
                                   BASE_ANGLE_MIN, BASE_ANGLE_MAX)
                shoulder_angle = clamp(shoulder_angle + shoulder_dir * step,
                                       SHOULDER_ANGLE_MIN, SHOULDER_ANGLE_MAX)
                needle_angle = clamp(needle_angle + needle_dir * step,
                                     NEEDLE_ANGLE_MIN, NEEDLE_ANGLE_MAX)

            # ── Write PWM ─────────────────────────────────────────────────
            base_pwm = angle_to_pwm(base_angle, BASE_ANGLE_MIN,
                                     BASE_ANGLE_MAX, BASE_PWM_MIN, BASE_PWM_MAX)
            shoulder_pwm = angle_to_pwm(shoulder_angle, SHOULDER_PHYSICAL_MIN,
                                         SHOULDER_PHYSICAL_MAX, SHOULDER_PWM_MIN,
                                         SHOULDER_PWM_MAX)
            needle_pwm = angle_to_pwm(needle_angle, NEEDLE_PHYSICAL_MIN,
                                       NEEDLE_PHYSICAL_MAX, NEEDLE_PWM_MIN,
                                       NEEDLE_PWM_MAX)

            pi.set_servo_pulsewidth(BASE_PIN, base_pwm)
            pi.set_servo_pulsewidth(SHOULDER_PIN, shoulder_pwm)
            pi.set_servo_pulsewidth(NEEDLE_PIN, needle_pwm)

            # ── Display ───────────────────────────────────────────────────
            stdscr.clear()
            stdscr.addstr(0, 0, "═══ SURGICAL ARM MANUAL CONTROL ═══")
            stdscr.addstr(2, 0, f"Velocity: {velocity:5.1f} °/sec  "
                                f"(max {VELOCITY_MAX}°/sec)")
            stdscr.addstr(3, 0, f"Loop:     {1.0/max(dt, 0.001):5.1f} Hz")
            if e_stop:
                stdscr.addstr(4, 0, "!! E-STOP ACTIVE !! (Space to resume)",
                              curses.A_REVERSE)
            stdscr.addstr(6, 0, "Joint            Angle      PWM     Keys")
            stdscr.addstr(7, 0, "─" * 50)
            stdscr.addstr(8, 0,
                f"Base yaw       {base_angle:+7.1f}°   {base_pwm:5d} µs   A / D")
            stdscr.addstr(9, 0,
                f"Shoulder pitch {shoulder_angle:+7.1f}°   {shoulder_pwm:5d} µs   W / S")
            stdscr.addstr(10, 0,
                f"Needle angle   {needle_angle:+7.1f}°   {needle_pwm:5d} µs   Q / E")
            stdscr.addstr(12, 0, "[ / ]  Adjust velocity    0  Home    "
                                 "Space  E-Stop    Esc  Quit")
            stdscr.refresh()

    finally:
        # Turn off all servo signals on exit
        pi.set_servo_pulsewidth(BASE_PIN, 0)
        pi.set_servo_pulsewidth(SHOULDER_PIN, 0)
        pi.set_servo_pulsewidth(NEEDLE_PIN, 0)
        pi.stop()


if __name__ == "__main__":
    curses.wrapper(main)
