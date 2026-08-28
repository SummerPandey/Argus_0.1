"""Headless end-to-end tests for nanoowl_monitor.main()'s capture loop.

These drive the real main() - real OpenCV drawing and color conversion,
the real AsyncOwlPredictor worker, the real monitor state machine - with
only the hardware edges faked: VideoCapture serves synthetic frames,
imshow/waitKey are scripted, and the predictor is a fake returning
NanoOWL-shaped detections. That pins down the loop wiring the threading
rework touched: results flow worker -> capture loop, detections arm the
session, R resets cleanly, Q quits cleanly, a mid-session inference
failure ends the session with a message instead of freezing, and a load
failure exits with the reason. It does not (and cannot, off-device)
validate the real TensorRT engine.
"""

import contextlib
import io
import threading
import time
import types
import unittest

import numpy as np

import nanoowl_test_stubs

nanoowl_test_stubs.install()

import cv2 as real_cv2

import nanoowl_monitor


def _detection(label_index, box, score=0.9):
    return types.SimpleNamespace(
        parent_id=0, box=box, scores=[score], labels=[label_index]
    )


def _two_hands_output():
    # Label indices follow the non-room PROMPT order: 0 = a left hand,
    # 1 = a right hand. Overlapping boxes, so the hands read as connected.
    return types.SimpleNamespace(
        detections=[
            _detection(0, (100.0, 100.0, 200.0, 200.0)),
            _detection(1, (190.0, 100.0, 290.0, 200.0)),
        ]
    )


class _FakeCapture:
    def __init__(self):
        self.frame = np.full((480, 640, 3), 90, dtype=np.uint8)

    def isOpened(self):
        return True

    def set(self, *args):
        return True

    def read(self):
        return True, self.frame.copy()

    def release(self):
        pass


class _ScriptedCv2:
    """Real cv2 for drawing/conversion; scripted camera, window, and
    keyboard so main() runs headless and deterministically."""

    def __init__(self, keys, idle_key=None):
        self._keys = list(keys)
        # What waitKey returns once the script runs out: Q ends a happy
        # path; 255 ("no key") forces error paths to end the loop
        # themselves, with a frame-count guard so a regression fails
        # instead of hanging the test run.
        self._idle_key = ord("q") if idle_key is None else idle_key
        self.frames_shown = 0
        self._frames_waited = 0

    def __getattr__(self, name):
        return getattr(real_cv2, name)

    def VideoCapture(self, index, backend=None):
        return _FakeCapture()

    def imshow(self, title, frame):
        self.frames_shown += 1

    def waitKey(self, delay):
        self._frames_waited += 1
        if self._frames_waited > 3000:
            raise AssertionError("main() loop did not stop; guard tripped")
        if self._keys:
            return self._keys.pop(0)
        return self._idle_key

    def destroyAllWindows(self):
        pass


class _HandsTreePredictor(nanoowl_test_stubs.TreePredictor):
    """Returns two connected hands every frame; optionally starts raising
    after a set number of predictions to simulate a mid-session
    CUDA/TensorRT failure."""

    fail_after = None

    def __init__(self, owl_predictor=None):
        super().__init__(owl_predictor)
        self.calls = 0

    def predict(self, image, tree, threshold, clip_text_encodings,
                owl_text_encodings):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("thermal meltdown")
        return _two_hands_output()


class _UnloadableTreePredictor(nanoowl_test_stubs.TreePredictor):
    def __init__(self, owl_predictor=None):
        raise RuntimeError("engine missing")


class MonitorLoopTests(unittest.TestCase):
    def _run_main(self, scripted_cv2, tree_predictor_class):
        originals = (nanoowl_monitor.cv2, nanoowl_monitor.TreePredictor)
        nanoowl_monitor.cv2 = scripted_cv2
        nanoowl_monitor.TreePredictor = tree_predictor_class
        self.addCleanup(
            lambda: setattr(nanoowl_monitor, "cv2", originals[0])
        )
        self.addCleanup(
            lambda: setattr(nanoowl_monitor, "TreePredictor", originals[1])
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            nanoowl_monitor.main()
        return output.getvalue()

    def _assert_worker_stopped(self):
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if not any(thread.name == "nanoowl-worker" and thread.is_alive()
                       for thread in threading.enumerate()):
                return
            time.sleep(0.01)
        self.fail("nanoowl-worker thread still alive after main() returned")

    def test_session_arms_resets_and_quits_cleanly(self):
        keys = [255] * 30 + [ord("r")] + [255] * 10 + [ord("q")]
        scripted = _ScriptedCv2(keys)

        class Predictor(_HandsTreePredictor):
            fail_after = None

        output = self._run_main(scripted, Predictor)
        self.assertIn("Session armed.", output)
        self.assertNotIn("failed", output)
        self.assertGreaterEqual(scripted.frames_shown, 40)
        self._assert_worker_stopped()

    def test_mid_session_inference_failure_ends_with_message(self):
        # No quit key is ever pressed: only the worker-health check can
        # end the loop, which is exactly what this test pins down.
        scripted = _ScriptedCv2([], idle_key=255)

        class Predictor(_HandsTreePredictor):
            fail_after = 3

        output = self._run_main(scripted, Predictor)
        self.assertIn("NanoOWL inference failed: thermal meltdown", output)
        self._assert_worker_stopped()

    def test_load_failure_exits_with_the_reason(self):
        scripted = _ScriptedCv2([], idle_key=255)
        with self.assertRaises(SystemExit) as caught:
            self._run_main(scripted, _UnloadableTreePredictor)
        self.assertIn("NanoOWL load failed", str(caught.exception))
        self.assertIn("engine missing", str(caught.exception))
        self._assert_worker_stopped()


if __name__ == "__main__":
    unittest.main()
