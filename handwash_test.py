# NOTE: this is the earliest prototype (no camera vision yet, driven by
# direct method calls). Superseded by handwash-next/bubbles/session.py's
# SessionEngine, which does the same job from real Observation data and has
# unit tests. Kept for reference. See the top-level README.md.

import cv2
import time
import numpy as np


# Use 5 seconds while testing at your desk.
# Change this to 20 for the finished prototype.
REQUIRED_RUB_TIME = 5

CAMERA_ID = 0


class HandwashMonitor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.session_started = False
        self.soap_used = False
        self.rub_start = None
        self.rubbing_seconds = 0.0
        self.rinsed = False
        self.dried = False

        self.result = None
        self.reason = ""
        self.result_time = 0

    def start_session(self):
        self.reset()
        self.session_started = True

    def use_soap(self):
        if self.session_started:
            self.soap_used = True

    def start_rubbing(self):
        if self.session_started and self.soap_used:
            if self.rub_start is None:
                self.rub_start = time.monotonic()

    def stop_rubbing(self):
        if self.rub_start is not None:
            elapsed = time.monotonic() - self.rub_start
            self.rubbing_seconds += elapsed
            self.rub_start = None

    def rinse(self):
        self.stop_rubbing()

        if (
            self.session_started
            and self.soap_used
            and self.rubbing_seconds >= REQUIRED_RUB_TIME
        ):
            self.rinsed = True

    def dry(self):
        if self.rinsed:
            self.dried = True

    def current_rub_time(self):
        total = self.rubbing_seconds

        if self.rub_start is not None:
            total += time.monotonic() - self.rub_start

        return total

    def end_session(self):
        self.stop_rubbing()

        if not self.session_started:
            return

        if not self.soap_used:
            self.fail("Soap was not detected")

        elif self.rubbing_seconds < REQUIRED_RUB_TIME:
            self.fail(
                f"Rubbing was too short: "
                f"{self.rubbing_seconds:.1f}/{REQUIRED_RUB_TIME}s"
            )

        elif not self.rinsed:
            self.fail("Rinsing was not detected")

        elif not self.dried:
            self.fail("Drying was not detected")

        else:
            self.result = "CORRECT"
            self.reason = "Handwashing sequence completed"
            self.result_time = time.monotonic()

    def fail(self, reason):
        self.result = "INCORRECT"
        self.reason = reason
        self.result_time = time.monotonic()


def add_status_panel(frame, monitor):
    height, width = frame.shape[:2]

    cv2.rectangle(
        frame,
        (10, 10),
        (min(width - 10, 580), 245),
        (25, 25, 25),
        -1
    )

    rub_time = monitor.current_rub_time()

    lines = [
        "HANDWASH MONITOR - DESK TEST",
        f"Session: {'ACTIVE' if monitor.session_started else 'WAITING'}",
        f"Soap: {'YES' if monitor.soap_used else 'NO'}",
        f"Rubbing: {rub_time:.1f}/{REQUIRED_RUB_TIME} seconds",
        f"Rinsed: {'YES' if monitor.rinsed else 'NO'}",
        f"Dried: {'YES' if monitor.dried else 'NO'}"
    ]

    for index, line in enumerate(lines):
        color = (255, 255, 255)

        if index == 0:
            color = (0, 255, 255)

        cv2.putText(
            frame,
            line,
            (25, 40 + index * 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

    return frame


def add_controls(frame):
    height, width = frame.shape[:2]

    controls = [
        "1 Start",
        "2 Soap",
        "3 Start rubbing",
        "4 Stop rubbing",
        "5 Rinse",
        "6 Dry",
        "E End session",
        "R Reset",
        "Q Quit"
    ]

    y = height - 20

    for text in reversed(controls):
        cv2.putText(
            frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1
        )
        y -= 24

    return frame


def warning_screen(frame, reason):
    red = np.zeros_like(frame)
    red[:] = (0, 0, 255)

    output = cv2.addWeighted(
        frame,
        0.20,
        red,
        0.80,
        0
    )

    cv2.putText(
        output,
        "HANDWASHING INCOMPLETE",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        3
    )

    cv2.putText(
        output,
        reason,
        (30, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    return output


def success_screen(frame):
    green = np.zeros_like(frame)
    green[:] = (0, 180, 0)

    output = cv2.addWeighted(
        frame,
        0.20,
        green,
        0.80,
        0
    )

    cv2.putText(
        output,
        "HANDWASHING COMPLETE",
        (30, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        3
    )

    return output


def main():
    monitor = HandwashMonitor()
    camera = cv2.VideoCapture(CAMERA_ID)

    if not camera.isOpened():
        print(f"Could not open camera /dev/video{CAMERA_ID}")
        print("Try changing CAMERA_ID to 1.")
        return

    print("Controls:")
    print("1 = start session")
    print("2 = use soap")
    print("3 = start rubbing timer")
    print("4 = stop rubbing timer")
    print("5 = rinse")
    print("6 = dry")
    print("e = end session")
    print("r = reset")
    print("q = quit")

    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read camera frame.")
            break

        frame = add_status_panel(frame, monitor)
        frame = add_controls(frame)

        # Keep the result screen visible for three seconds.
        if monitor.result is not None:
            elapsed = time.monotonic() - monitor.result_time

            if elapsed <= 3:
                if monitor.result == "CORRECT":
                    frame = success_screen(frame)
                else:
                    frame = warning_screen(frame, monitor.reason)
            else:
                monitor.reset()

        cv2.imshow("Handwash Monitor", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("1"):
            monitor.start_session()

        elif key == ord("2"):
            monitor.use_soap()

        elif key == ord("3"):
            monitor.start_rubbing()

        elif key == ord("4"):
            monitor.stop_rubbing()

        elif key == ord("5"):
            monitor.rinse()

        elif key == ord("6"):
            monitor.dry()

        elif key == ord("e"):
            monitor.end_session()

        elif key == ord("r"):
            monitor.reset()

        elif key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
