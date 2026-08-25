from __future__ import annotations

import argparse
import queue
import threading
import time

from .session import Observation, SessionEngine, Stage
from .vision import HandVision, VisionResult


class LatestFrameCamera:
    def __init__(self, cv2, camera_id=None, width=640, height=480):
        self.capture, selected = None, None
        candidates = (camera_id,) if camera_id is not None else range(4)
        for candidate in candidates:
            capture = cv2.VideoCapture(candidate)
            if capture.isOpened():
                self.capture, selected = capture, candidate
                break
            capture.release()
        if self.capture is None:
            raise RuntimeError(
                "No camera could be opened. Reconnect the USB camera and close "
                "other apps using it, then launch Argus again."
            )
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.camera_id = selected
        self.frames, self.running, self.error = queue.Queue(maxsize=1), True, None
        threading.Thread(target=self._capture, daemon=True).start()

    def _capture(self):
        while self.running:
            ok, frame = self.capture.read()
            if not ok:
                self.error, self.running = "Camera stopped returning frames", False
                break
            try:
                self.frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frames.put_nowait(frame)
            except queue.Full:
                pass

    def latest(self, timeout=1):
        return self.frames.get(timeout=timeout)

    def close(self):
        self.running = False
        self.capture.release()


class AsyncHandVision:
    """Run landmark inference off the display thread and drop stale frames."""

    def __init__(self, mp, cv2, np, inference_width=448, max_inference_fps=15,
                 detect_foam=True):
        self.cv2, self.np = cv2, np
        self.inference_width = inference_width
        self.minimum_interval = 1.0 / max_inference_fps
        self.last_submit = 0.0
        self.detect_foam = detect_foam
        self.vision = HandVision(mp)
        self.jobs = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.result = VisionResult(Observation(0), tuple(), 0, 0, 0, 0)
        self.sequence = 0
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, frame, timestamp):
        if timestamp - self.last_submit < self.minimum_interval:
            return
        self.last_submit = timestamp
        height, width = frame.shape[:2]
        scaled_height = max(1, int(height * self.inference_width / width))
        small = self.cv2.resize(frame, (self.inference_width, scaled_height),
                                interpolation=self.cv2.INTER_AREA)
        try:
            self.jobs.get_nowait()
        except queue.Empty:
            pass
        try:
            self.jobs.put_nowait((small, timestamp))
        except queue.Full:
            pass

    def latest(self):
        with self.lock:
            return self.sequence, self.result

    def _run(self):
        while self.running:
            try:
                frame, timestamp = self.jobs.get(timeout=.2)
            except queue.Empty:
                continue
            result = self.vision.process(frame, timestamp, self.cv2, self.np,
                                         detect_foam=self.detect_foam)
            with self.lock:
                self.result = result
                self.sequence += 1

    def close(self):
        self.running = False
        self.thread.join(timeout=2)
        self.vision.close()


def draw_landmarks(frame, result, cv2):
    height, width = frame.shape[:2]
    links = ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),
             (9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
             (13,17),(17,18),(18,19),(19,20),(0,17))
    for hand in result.landmarks:
        pixels = [(int(x*width), int(y*height)) for x, y in hand]
        for a, b in links:
            cv2.line(frame, pixels[a], pixels[b], (94, 224, 189), 2, cv2.LINE_AA)
        for point in pixels:
            cv2.circle(frame, point, 3, (255, 255, 255), -1, cv2.LINE_AA)
    for elbow, wrist in result.forearms:
        a = (int(elbow[0]*width), int(elbow[1]*height))
        b = (int(wrist[0]*width), int(wrist[1]*height))
        cv2.line(frame, a, b, (232, 199, 105), 5, cv2.LINE_AA)
        cv2.circle(frame, a, 6, (232, 199, 105), -1, cv2.LINE_AA)
        cv2.circle(frame, b, 7, (167, 208, 69), -1, cv2.LINE_AA)


def draw_panel(frame, state, vision, fps, auto_soap, room_mode, debug, cv2):
    height, width = frame.shape[:2]
    right = min(width-14, 626)
    overlay = frame.copy()
    cv2.rectangle(overlay, (14, 14), (right, 173), (13, 39, 43), -1)
    cv2.addWeighted(overlay, .88, frame, .12, 0, frame)

    # Product mark and session mode.
    cv2.circle(frame, (34, 35), 7, (167, 208, 69), -1, cv2.LINE_AA)
    cv2.circle(frame, (43, 29), 4, (232, 199, 105), -1, cv2.LINE_AA)
    cv2.putText(frame, "ARGUS", (55, 40), cv2.FONT_HERSHEY_SIMPLEX,
                .46, (235, 248, 244), 1, cv2.LINE_AA)
    mode = "ROOM TEST" if room_mode else "SINK SESSION"
    mode_width = 104 if room_mode else 122
    cv2.rectangle(frame, (right-mode_width-14, 24), (right-14, 47), (32, 74, 76), -1)
    cv2.putText(frame, mode, (right-mode_width-4, 40), cv2.FONT_HERSHEY_SIMPLEX,
                .36, (185, 216, 211), 1, cv2.LINE_AA)

    title_color = (167, 235, 92) if state.stage == Stage.COMPLETE else (255, 255, 255)
    cv2.putText(frame, state.message, (30, 76), cv2.FONT_HERSHEY_SIMPLEX,
                .68, title_color, 2, cv2.LINE_AA)

    # Calm progress bar with timer aligned to its right edge.
    bar_left, bar_right, bar_y = 30, right-112, 99
    cv2.rectangle(frame, (bar_left, bar_y), (bar_right, bar_y+10), (46, 77, 79), -1)
    fill = int((bar_right-bar_left) * state.progress)
    if fill:
        cv2.rectangle(frame, (bar_left, bar_y), (bar_left+fill, bar_y+10),
                      (167, 208, 69), -1)
    cv2.putText(frame, f"{state.rubbing_seconds:04.1f} / 20s", (right-98, bar_y+10),
                cv2.FONT_HERSHEY_SIMPLEX, .42, (220, 236, 232), 1, cv2.LINE_AA)

    hands = "HANDS READY" if state.hands_present else "FINDING HANDS"
    soap = "SOAP BYPASSED" if room_mode else ("SOAP SEEN" if state.soap_seen else "SOAP NEEDED")
    cv2.circle(frame, (34, 135), 5,
               (167, 208, 69) if state.hands_present else (120, 135, 135), -1)
    cv2.putText(frame, hands, (47, 140), cv2.FONT_HERSHEY_SIMPLEX,
                .38, (202, 221, 217), 1, cv2.LINE_AA)
    cv2.circle(frame, (181, 135), 5,
               (167, 208, 69) if (state.soap_seen or room_mode) else (120, 135, 135), -1)
    cv2.putText(frame, soap, (194, 140), cv2.FONT_HERSHEY_SIMPLEX,
                .38, (202, 221, 217), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:02.0f} FPS", (right-74, 140), cv2.FONT_HERSHEY_SIMPLEX,
                .38, (135, 166, 164), 1, cv2.LINE_AA)

    controls = "R  RESET     D  DETAILS     Q  QUIT"
    if not room_mode:
        controls = "S  SOAP     A  AUTO     " + controls
    cv2.putText(frame, controls, (30, 162), cv2.FONT_HERSHEY_SIMPLEX,
                .31, (125, 157, 155), 1, cv2.LINE_AA)

    if debug:
        cv2.rectangle(frame, (14, height-50), (right, height-14), (13, 39, 43), -1)
        details = (f"hand {vision.contact:.2f}   forearm {vision.forearm_contact:.2f}   "
                   f"motion {vision.motion:.2f}   reversals {vision.reversals}   "
                   f"foam {vision.soap_signal:.2f}")
        cv2.putText(frame, details, (29, height-27), cv2.FONT_HERSHEY_SIMPLEX,
                    .39, (180, 207, 203), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--room", action="store_true",
                        help="Track real rubbing but bypass soap and water")
    args = parser.parse_args()
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
    except ImportError as exc:
        raise SystemExit("Live mode needs MediaPipe. Run: ./install-camera-deps.sh") from exc

    try:
        camera = LatestFrameCamera(cv2)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    vision = AsyncHandVision(mp, cv2, np, detect_foam=not args.room)
    engine = SessionEngine(required_rub_seconds=20, confirmation_seconds=1.5,
                           require_soap=not args.room)
    manual_soap_until, auto_soap = 0.0, True
    vision_sequence = -1
    result = VisionResult(Observation(0), tuple(), 0, 0, 0, 0)
    state = engine.snapshot(False)
    fps, frames, fps_at, debug = 0.0, 0, time.monotonic(), False
    try:
        while camera.running:
            frame, now = camera.latest(), time.monotonic()
            vision.submit(frame, now)
            new_sequence, newest = vision.latest()
            if new_sequence != vision_sequence:
                vision_sequence, result = new_sequence, newest
                soap = result.observation.soap_confidence if auto_soap and not args.room else 0.0
                soap = 1.0 if now < manual_soap_until else soap
                state = engine.update(Observation(result.observation.timestamp,
                                                  result.observation.hands_confidence,
                                                  result.observation.rubbing_confidence, soap))
            frames += 1
            if now-fps_at >= 1:
                fps, frames, fps_at = frames/(now-fps_at), 0, now
            draw_landmarks(frame, result, cv2)
            draw_panel(frame, state, result, fps, auto_soap, args.room, debug, cv2)
            cv2.imshow("Argus — Live Hand Hygiene", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                engine.reset()
            if key == ord("s") and not args.room:
                manual_soap_until = now+1
            if key == ord("a") and not args.room:
                auto_soap = not auto_soap
            if key == ord("d"):
                debug = not debug
    finally:
        vision.close()
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
