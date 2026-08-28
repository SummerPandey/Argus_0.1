"""Failure-propagation tests for bubbles/camera.py's worker threads.

LatestFrameCamera and AsyncHandVision each run a daemon thread; if that
thread dies, the failure must land on .error with .running cleared so
the main loop can report it and exit - never a silent freeze on the last
result. These tests drive both paths with fakes (no real camera or
MediaPipe needed): a capture whose read() raises or stops, and a vision
worker whose process() raises. bubbles/camera.py imports cleanly without
cv2/mediapipe because main() imports them lazily, which is what makes
this testable anywhere.
"""

import queue
import time
import unittest

from bubbles import camera as camera_module
from bubbles.camera import AsyncHandVision, LatestFrameCamera


def _wait_for(condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return condition()


class _FakeCapture:
    """Stands in for cv2.VideoCapture: opens successfully, then read()
    either raises or reports a stopped stream."""

    def __init__(self, raises=False):
        self.raises = raises
        self.released = False

    def isOpened(self):
        return True

    def set(self, *args):
        return True

    def read(self):
        if self.raises:
            raise RuntimeError("USB camera vanished")
        return False, None

    def release(self):
        self.released = True


class _FakeCv2:
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FOURCC = 6
    CAP_PROP_BUFFERSIZE = 38

    def __init__(self, capture):
        self._capture = capture

    def VideoCapture(self, candidate):
        return self._capture

    @staticmethod
    def VideoWriter_fourcc(*characters):
        return 0


class LatestFrameCameraFailureTests(unittest.TestCase):
    def test_raising_read_sets_error_instead_of_dying_silently(self):
        capture = _FakeCapture(raises=True)
        camera = LatestFrameCamera(_FakeCv2(capture), camera_id=0)
        self.assertTrue(_wait_for(lambda: not camera.running))
        self.assertIn("USB camera vanished", camera.error)
        camera.close()  # must still be safe after the thread died
        self.assertTrue(capture.released)

    def test_stopped_stream_sets_error(self):
        camera = LatestFrameCamera(_FakeCv2(_FakeCapture()), camera_id=0)
        self.assertTrue(_wait_for(lambda: not camera.running))
        self.assertEqual(camera.error, "Camera stopped returning frames")
        camera.close()


class _ExplodingVision:
    def __init__(self, mp):
        self.closed = False

    def process(self, frame, timestamp, cv2, np, detect_foam=True):
        raise RuntimeError("mediapipe timestamp went backwards")

    def close(self):
        self.closed = True


class _CountingVision:
    def __init__(self, mp):
        self.closed = False

    def process(self, frame, timestamp, cv2, np, detect_foam=True):
        return ("result", timestamp)

    def close(self):
        self.closed = True


class AsyncHandVisionFailureTests(unittest.TestCase):
    def _make_vision(self, fake_class):
        original = camera_module.HandVision
        camera_module.HandVision = fake_class
        self.addCleanup(setattr, camera_module, "HandVision", original)
        return AsyncHandVision(mp=None, cv2=None, np=None)

    def test_process_exception_sets_error_and_stops_worker(self):
        vision = self._make_vision(_ExplodingVision)
        # Feed the worker directly; submit() needs cv2 for its resize.
        vision.jobs.put((object(), 0.0))
        self.assertTrue(_wait_for(lambda: vision.error is not None))
        self.assertIn("mediapipe timestamp went backwards", vision.error)
        self.assertFalse(vision.running)
        vision.close()
        self.assertFalse(vision.thread.is_alive())
        self.assertTrue(vision.vision.closed)

    def test_results_still_publish_when_processing_succeeds(self):
        vision = self._make_vision(_CountingVision)
        try:
            vision.jobs.put((object(), 1.0))
            self.assertTrue(_wait_for(lambda: vision.latest()[0] == 1))
            sequence, result = vision.latest()
            self.assertEqual(result, ("result", 1.0))
            self.assertIsNone(vision.error)

            vision.jobs.put((object(), 2.0))
            self.assertTrue(_wait_for(lambda: vision.latest()[0] == 2))
            _, result = vision.latest()
            self.assertEqual(result, ("result", 2.0))
        finally:
            vision.close()


if __name__ == "__main__":
    unittest.main()
