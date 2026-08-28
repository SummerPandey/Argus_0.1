"""Threading-mechanics tests for nanoowl_monitor's AsyncOwlPredictor.

The real NanoOWL/TensorRT stack only exists inside the Jetson container,
so the `nanoowl` package (and PIL, when absent) is stubbed just enough to
import nanoowl_monitor, and predictions run through a fake predictor.
What these tests pin down is the part of the threading that is
hardware-independent and must never regress:

- exactly one thread ever touches the predictor, loading included (the
  CUDA/TensorRT context-ownership rule), and it is never the caller;
- stale frames are dropped rather than queued when inference can't keep
  up with submission;
- the submitted frame is isolated from the capture loop's in-place HUD
  drawing;
- a failure while loading or predicting surfaces on .error instead of
  killing the worker silently;
- close() returns promptly and actually stops the thread.

None of this substitutes for running against the real engine on-device -
it guards the mechanics around that call, not the call itself.
"""

import threading
import time
import types
import unittest

import numpy as np

import nanoowl_test_stubs

nanoowl_test_stubs.install()

import nanoowl_monitor


def _frame(value=0):
    return np.full((24, 32, 3), value, dtype=np.uint8)


def _wait_for(condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return condition()


class FakePredictor:
    """Records every thread that calls predict(), optionally slow or
    failing, so tests can assert the single-owner-thread rule."""

    def __init__(self, delay=0.0, fail=False):
        self.delay = delay
        self.fail = fail
        self.calls = 0
        self.caller_threads = set()
        self.lock = threading.Lock()

    def predict(self, image, tree, threshold, clip_text_encodings,
                owl_text_encodings):
        with self.lock:
            self.calls += 1
            self.caller_threads.add(threading.get_ident())
        if self.fail:
            raise RuntimeError("simulated TensorRT failure")
        if self.delay:
            time.sleep(self.delay)
        return types.SimpleNamespace(detections=[])


def _make_owl(predictor, loader_threads=None, max_inference_fps=10_000):
    def loader():
        if loader_threads is not None:
            loader_threads.add(threading.get_ident())
        return predictor, "clip-encodings", "owl-encodings"

    # A very high default fps cap keeps these mechanics tests about the
    # queue/thread behavior; the cap itself has its own dedicated test.
    return nanoowl_monitor.AsyncOwlPredictor(
        loader, tree=object(), threshold=0.1,
        max_inference_fps=max_inference_fps,
    )


class AsyncOwlPredictorTests(unittest.TestCase):
    def test_latest_is_empty_before_any_result(self):
        owl = _make_owl(FakePredictor())
        try:
            self.assertTrue(owl.wait_ready(timeout=3))
            sequence, result = owl.latest()
            self.assertEqual(sequence, 0)
            self.assertIsNone(result)
            self.assertIsNone(owl.error)
        finally:
            owl.close()

    def test_loading_and_predicting_share_one_worker_thread(self):
        loader_threads = set()
        predictor = FakePredictor()
        owl = _make_owl(predictor, loader_threads)
        try:
            self.assertTrue(owl.wait_ready(timeout=3))
            owl.submit(_frame(), time.monotonic())
            self.assertTrue(_wait_for(lambda: owl.latest()[1] is not None))
        finally:
            owl.close()
        self.assertEqual(len(predictor.caller_threads), 1)
        self.assertEqual(loader_threads, predictor.caller_threads)
        self.assertNotIn(threading.get_ident(), predictor.caller_threads)

    def test_flooded_submissions_drop_frames_instead_of_queueing(self):
        predictor = FakePredictor(delay=0.03)
        owl = _make_owl(predictor)
        try:
            self.assertTrue(owl.wait_ready(timeout=3))
            submitted = 200
            for index in range(submitted):
                owl.submit(_frame(index % 255), time.monotonic())
                time.sleep(0.001)
            self.assertTrue(_wait_for(lambda: owl.latest()[1] is not None))
        finally:
            owl.close()
        self.assertGreaterEqual(predictor.calls, 1)
        self.assertLess(predictor.calls, submitted)

    def test_submitted_frame_is_isolated_from_caller_drawing(self):
        owl = _make_owl(FakePredictor())
        try:
            self.assertTrue(owl.wait_ready(timeout=3))
            frame = _frame(7)
            pristine = frame.copy()
            owl.submit(frame, time.monotonic())
            frame[:] = 255  # the capture loop draws the HUD in place
            self.assertTrue(_wait_for(lambda: owl.latest()[1] is not None))
            _, result = owl.latest()
            self.assertTrue(np.array_equal(result.frame, pristine))
        finally:
            owl.close()

    def test_loader_failure_surfaces_on_error(self):
        def loader():
            raise RuntimeError("engine file missing")

        owl = nanoowl_monitor.AsyncOwlPredictor(loader, tree=object(), threshold=0.1)
        self.assertFalse(owl.wait_ready(timeout=3))
        self.assertIn("engine file missing", owl.error)
        owl.close()
        self.assertFalse(owl.thread.is_alive())

    def test_prediction_failure_surfaces_on_error(self):
        owl = _make_owl(FakePredictor(fail=True))
        try:
            self.assertTrue(owl.wait_ready(timeout=3))
            owl.submit(_frame(), time.monotonic())
            self.assertTrue(_wait_for(lambda: owl.error is not None))
            self.assertIn("simulated TensorRT failure", owl.error)
            self.assertFalse(owl.running)
        finally:
            owl.close()
        self.assertFalse(owl.thread.is_alive())

    def test_inference_cadence_cap_skips_too_soon_submissions(self):
        # Deterministic: the cap compares the caller-supplied timestamps,
        # so fake ones drive it exactly. At 5 FPS the minimum interval is
        # 0.2 "seconds"; a submission 0.05 after the first must be
        # skipped, one 0.3 after must go through.
        predictor = FakePredictor()
        owl = _make_owl(predictor, max_inference_fps=5)
        try:
            self.assertTrue(owl.wait_ready(timeout=3))
            owl.submit(_frame(), 10.0)
            self.assertTrue(_wait_for(lambda: predictor.calls == 1))
            owl.submit(_frame(), 10.05)  # inside the interval: skipped
            time.sleep(0.05)
            self.assertEqual(predictor.calls, 1)
            self.assertTrue(owl.jobs.empty())
            owl.submit(_frame(), 10.3)  # past the interval: accepted
            self.assertTrue(_wait_for(lambda: predictor.calls == 2))
        finally:
            owl.close()

    def test_close_returns_promptly_when_idle(self):
        owl = _make_owl(FakePredictor())
        self.assertTrue(owl.wait_ready(timeout=3))
        started = time.monotonic()
        owl.close()
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertFalse(owl.thread.is_alive())


if __name__ == "__main__":
    unittest.main()
