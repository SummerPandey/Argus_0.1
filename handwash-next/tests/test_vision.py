import unittest
from bubbles.vision import FoamSignal, LocalMotionSignal, RubbingSignal, clamp, distance, palm_center, palm_scale


def hand(cx, cy, scale=.1):
    points = [(cx, cy)] * 21
    points[0], points[5] = (cx, cy+scale), (cx-scale*.55, cy)
    points[9], points[13], points[17] = (cx, cy-scale), (cx+scale*.3, cy), (cx+scale*.55, cy)
    return points


class VisionTests(unittest.TestCase):
    def test_distant_hands_do_not_rub(self):
        signal = RubbingSignal()
        for i in range(20):
            confidence, contact, _, _ = signal.update(hand(.2+i*.002, .5), hand(.8, .5))
        self.assertLess(contact, .1)
        self.assertLess(confidence, .1)

    def test_alternating_close_motion_builds_confidence(self):
        signal, peak = RubbingSignal(), 0
        for i in range(30):
            offset = .035 if i % 2 else -.035
            confidence, _, _, reversals = signal.update(hand(.47+offset, .5), hand(.53-offset, .5))
            peak = max(peak, confidence)
        self.assertGreaterEqual(reversals, 3)
        self.assertGreater(peak, .5)

    def test_local_crop_motion_reacts_to_internal_change(self):
        import cv2
        import numpy as np
        signal = LocalMotionSignal()
        points = (hand(.5, .5, .2),)
        first = np.zeros((240, 320, 3), dtype=np.uint8)
        second = first.copy()
        cv2.rectangle(second, (130, 90), (190, 150), (255, 255, 255), -1)
        self.assertEqual(signal.update(first, points, cv2, np), 0)
        self.assertGreater(signal.update(second, points, cv2, np), .5)


class GeometryHelperTests(unittest.TestCase):
    def test_clamp_bounds_to_unit_range(self):
        self.assertEqual(clamp(-.5), 0.0)
        self.assertEqual(clamp(1.5), 1.0)
        self.assertEqual(clamp(.3), .3)

    def test_distance_is_euclidean(self):
        self.assertAlmostEqual(distance((0, 0), (3, 4)), 5.0)

    def test_palm_center_and_scale_of_a_flat_hand(self):
        points = hand(.5, .5, .2)
        cx, cy = palm_center(points)
        # The hand() fixture's thumb/pinky offsets aren't symmetric, so the
        # center sits close to (cx, cy) but not exactly on it.
        self.assertAlmostEqual(cx, .5, places=1)
        self.assertAlmostEqual(cy, .5, places=3)
        self.assertGreater(palm_scale(points), 0)


class FoamSignalTests(unittest.TestCase):
    def test_no_signal_without_a_bright_low_saturation_region(self):
        import cv2
        import numpy as np
        signal = FoamSignal()
        points = (hand(.5, .5, .2),)
        dark_frame = np.full((240, 320, 3), 40, dtype=np.uint8)
        score = 0.0
        for _ in range(12):
            score = signal.update(dark_frame, points, cv2, np)
        self.assertEqual(score, 0.0)

    def test_sustained_bright_low_saturation_region_registers_foam(self):
        import cv2
        import numpy as np
        signal = FoamSignal()
        points = (hand(.5, .5, .2),)
        # Establish a dark baseline first, matching how the real camera
        # loop calls update() every frame before soap ever appears.
        baseline_frame = np.full((240, 320, 3), 40, dtype=np.uint8)
        for _ in range(5):
            signal.update(baseline_frame, points, cv2, np)
        # A near-white, low-saturation frame mimics bright foam filling
        # the hand ROI. FoamSignal requires it to be sustained across
        # several consecutive frames before reporting anything.
        foam_frame = np.full((240, 320, 3), 245, dtype=np.uint8)
        scores = [signal.update(foam_frame, points, cv2, np) for _ in range(12)]
        self.assertEqual(scores[0], 0.0)
        self.assertGreater(scores[-1], 0.0)

    def test_empty_roi_returns_zero(self):
        import cv2
        import numpy as np
        signal = FoamSignal()
        # Hand landmarks entirely outside the frame collapse the ROI to
        # nothing; update() must not crash on a zero-size crop.
        points = (hand(-5, -5, .01),)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        self.assertEqual(signal.update(frame, points, cv2, np), 0.0)


if __name__ == "__main__":
    unittest.main()
