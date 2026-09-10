import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import equipment_logic
from equipment_logic import (
    BLOOD_STAIN_RATIO_THRESHOLD,
    CONTAMINATION_CONFIRMATION_SECONDS,
    DIRT_STAIN_RATIO_THRESHOLD,
    ITEM_ABSENCE_GRACE_SECONDS,
    ITEM_CONFIRMATION_SECONDS,
    TUNABLE_DEFAULTS,
    advance_item,
    advance_scan,
    best_match_for,
    blood_stain_ratio,
    contaminated_items,
    crop_box,
    dirt_stain_ratio,
    load_config,
    match_checklist,
    matches_required,
    missing_items,
    reset_scan,
    scan_passed,
)


def solid_hsv_crop(hue, saturation, value, size=20):
    """A uint8 BGR image where every pixel has the given HSV values,
    built by converting HSV->BGR so tests don't depend on the reader's
    mental math about what BGR triplet lands where in HSV space."""
    hsv = np.full((size, size, 3), (hue, saturation, value), dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class CropBoxTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((100, 200, 3), dtype=np.uint8)

    def test_crops_within_bounds(self):
        crop = crop_box(self.frame, (10, 10, 30, 40))
        self.assertEqual(crop.shape, (30, 20, 3))

    def test_clamps_box_extending_past_frame(self):
        crop = crop_box(self.frame, (-10, -10, 20, 20))
        self.assertEqual(crop.shape, (20, 20, 3))

    def test_returns_none_for_fully_offframe_box(self):
        self.assertIsNone(crop_box(self.frame, (300, 300, 320, 320)))

    def test_returns_none_for_degenerate_box(self):
        self.assertIsNone(crop_box(self.frame, (10, 10, 10, 10)))


class ContaminationRatioTests(unittest.TestCase):
    def test_pure_blood_red_scores_high_blood_ratio(self):
        crop = solid_hsv_crop(hue=3, saturation=180, value=90)
        self.assertGreater(blood_stain_ratio(crop), 0.9)

    def test_clean_bright_steel_scores_low_on_both(self):
        # Bright, low-saturation, neutral hue - what a clean stainless
        # instrument under normal light looks like.
        crop = solid_hsv_crop(hue=0, saturation=10, value=220)
        self.assertEqual(blood_stain_ratio(crop), 0.0)
        self.assertEqual(dirt_stain_ratio(crop), 0.0)

    def test_brown_grime_scores_high_dirt_ratio(self):
        crop = solid_hsv_crop(hue=20, saturation=120, value=80)
        self.assertGreater(dirt_stain_ratio(crop), 0.9)

    def test_missing_crop_scores_zero(self):
        self.assertEqual(blood_stain_ratio(None), 0.0)
        self.assertEqual(dirt_stain_ratio(np.zeros((0, 0, 3), dtype=np.uint8)), 0.0)

    def test_partial_stain_gives_partial_ratio(self):
        crop = solid_hsv_crop(hue=0, saturation=10, value=220, size=10)
        crop[:5, :, :] = solid_hsv_crop(hue=3, saturation=180, value=90, size=10)[:5, :, :]
        ratio = blood_stain_ratio(crop)
        self.assertAlmostEqual(ratio, 0.5, delta=0.02)


class ChecklistMatchingTests(unittest.TestCase):
    def test_matches_loose_label_phrasing(self):
        self.assertTrue(matches_required("a needle holder", "needle holder"))
        self.assertTrue(matches_required("forceps", "a pair of forceps"))
        self.assertFalse(matches_required("a retractor", "scalpel"))

    def test_best_match_picks_highest_score(self):
        detections = [
            {"label": "a scalpel", "score": 0.4, "box": (0, 0, 10, 10)},
            {"label": "a scalpel", "score": 0.8, "box": (20, 20, 30, 30)},
            {"label": "forceps", "score": 0.9, "box": (40, 40, 50, 50)},
        ]
        match = best_match_for("scalpel", detections)
        self.assertEqual(match["score"], 0.8)

    def test_best_match_is_none_when_absent(self):
        detections = [{"label": "forceps", "score": 0.9, "box": (0, 0, 10, 10)}]
        self.assertIsNone(best_match_for("scalpel", detections))

    def test_match_checklist_covers_every_required_name(self):
        detections = [{"label": "a scalpel", "score": 0.5, "box": (0, 0, 10, 10)}]
        matches = match_checklist(detections, required=["scalpel", "scissors"])
        self.assertIsNotNone(matches["scalpel"])
        self.assertIsNone(matches["scissors"])


class ResetScanTests(unittest.TestCase):
    def test_starts_with_nothing_confirmed_or_flagged(self):
        scan = reset_scan(required=["scalpel", "forceps"])
        self.assertEqual(set(scan["items"]), {"scalpel", "forceps"})
        self.assertEqual(missing_items(scan), ["scalpel", "forceps"])
        self.assertEqual(contaminated_items(scan), [])
        self.assertFalse(scan_passed(scan))


class AdvanceItemTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.clean_box = (10, 10, 30, 30)
        self.clean_frame = self.frame.copy()
        self.clean_frame[10:30, 10:30] = solid_hsv_crop(0, 10, 220, size=20)

    def test_confirms_present_after_confirmation_seconds(self):
        item = reset_scan(required=["scalpel"])["items"]["scalpel"]
        detection = {"label": "a scalpel", "score": 0.5, "box": self.clean_box}
        advance_item(item, detection, self.clean_frame, dt=ITEM_CONFIRMATION_SECONDS - 0.1)
        self.assertFalse(item["confirmed"])
        advance_item(item, detection, self.clean_frame, dt=0.2)
        self.assertTrue(item["confirmed"])

    def test_never_seen_stays_missing(self):
        item = reset_scan(required=["scalpel"])["items"]["scalpel"]
        advance_item(item, None, self.clean_frame, dt=5.0)
        self.assertFalse(item["confirmed"])

    def test_brief_absence_after_confirmation_is_tolerated(self):
        item = reset_scan(required=["scalpel"])["items"]["scalpel"]
        detection = {"label": "a scalpel", "score": 0.5, "box": self.clean_box}
        advance_item(item, detection, self.clean_frame, dt=ITEM_CONFIRMATION_SECONDS)
        self.assertTrue(item["confirmed"])
        advance_item(item, None, self.clean_frame, dt=ITEM_ABSENCE_GRACE_SECONDS - 0.1)
        self.assertTrue(item["confirmed"])

    def test_prolonged_absence_after_confirmation_clears_it(self):
        item = reset_scan(required=["scalpel"])["items"]["scalpel"]
        detection = {"label": "a scalpel", "score": 0.5, "box": self.clean_box}
        advance_item(item, detection, self.clean_frame, dt=ITEM_CONFIRMATION_SECONDS)
        self.assertTrue(item["confirmed"])
        advance_item(item, None, self.clean_frame, dt=ITEM_ABSENCE_GRACE_SECONDS)
        self.assertFalse(item["confirmed"])

    def test_bloodstained_item_latches_blood_flag(self):
        item = reset_scan(required=["scalpel"])["items"]["scalpel"]
        frame = self.frame.copy()
        frame[10:30, 10:30] = solid_hsv_crop(3, 180, 90, size=20)
        detection = {"label": "a scalpel", "score": 0.5, "box": self.clean_box}
        advance_item(item, detection, frame, dt=CONTAMINATION_CONFIRMATION_SECONDS)
        self.assertTrue(item["blood_flagged"])
        self.assertFalse(item["dirt_flagged"])

    def test_dirty_item_latches_dirt_flag(self):
        item = reset_scan(required=["scalpel"])["items"]["scalpel"]
        frame = self.frame.copy()
        frame[10:30, 10:30] = solid_hsv_crop(20, 120, 80, size=20)
        detection = {"label": "a scalpel", "score": 0.5, "box": self.clean_box}
        advance_item(item, detection, frame, dt=CONTAMINATION_CONFIRMATION_SECONDS)
        self.assertTrue(item["dirt_flagged"])
        self.assertFalse(item["blood_flagged"])

    def test_clean_item_never_flags(self):
        item = reset_scan(required=["scalpel"])["items"]["scalpel"]
        detection = {"label": "a scalpel", "score": 0.5, "box": self.clean_box}
        advance_item(item, detection, self.clean_frame, dt=5.0)
        self.assertFalse(item["blood_flagged"])
        self.assertFalse(item["dirt_flagged"])

    def test_flag_stays_latched_once_the_stain_goes_out_of_view(self):
        # A flag is sticky by design - see advance_item's docstring - so a
        # later clean-looking frame (glare, a hand passing over it) can't
        # quietly clear a contamination reading that was already confirmed.
        item = reset_scan(required=["scalpel"])["items"]["scalpel"]
        bloody_frame = self.frame.copy()
        bloody_frame[10:30, 10:30] = solid_hsv_crop(3, 180, 90, size=20)
        detection = {"label": "a scalpel", "score": 0.5, "box": self.clean_box}
        advance_item(item, detection, bloody_frame, dt=CONTAMINATION_CONFIRMATION_SECONDS)
        self.assertTrue(item["blood_flagged"])
        advance_item(item, detection, self.clean_frame, dt=5.0)
        self.assertTrue(item["blood_flagged"])

    def test_brief_stain_reading_below_confirmation_time_does_not_latch(self):
        item = reset_scan(required=["scalpel"])["items"]["scalpel"]
        bloody_frame = self.frame.copy()
        bloody_frame[10:30, 10:30] = solid_hsv_crop(3, 180, 90, size=20)
        detection = {"label": "a scalpel", "score": 0.5, "box": self.clean_box}
        advance_item(item, detection, bloody_frame, dt=CONTAMINATION_CONFIRMATION_SECONDS - 0.1)
        self.assertFalse(item["blood_flagged"])


class AdvanceScanTests(unittest.TestCase):
    def test_full_pass_once_every_item_confirmed_and_clean(self):
        scan = reset_scan(required=["scalpel", "forceps"])
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:] = solid_hsv_crop(0, 10, 220, size=1)
        matches = {
            "scalpel": {"label": "a scalpel", "score": 0.6, "box": (0, 0, 10, 10)},
            "forceps": {"label": "forceps", "score": 0.6, "box": (20, 20, 30, 30)},
        }
        for _ in range(3):
            advance_scan(scan, matches, frame, dt=ITEM_CONFIRMATION_SECONDS)
        self.assertTrue(scan_passed(scan))
        self.assertEqual(missing_items(scan), [])
        self.assertEqual(contaminated_items(scan), [])

    def test_missing_item_fails_the_scan(self):
        scan = reset_scan(required=["scalpel", "forceps"])
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        matches = {
            "scalpel": {"label": "a scalpel", "score": 0.6, "box": (0, 0, 10, 10)},
            "forceps": None,
        }
        advance_scan(scan, matches, frame, dt=ITEM_CONFIRMATION_SECONDS)
        self.assertEqual(missing_items(scan), ["forceps"])
        self.assertFalse(scan_passed(scan))

    def test_contaminated_item_fails_the_scan_even_if_present(self):
        scan = reset_scan(required=["scalpel"])
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[0:10, 0:10] = solid_hsv_crop(3, 180, 90, size=10)
        matches = {"scalpel": {"label": "a scalpel", "score": 0.6, "box": (0, 0, 10, 10)}}
        advance_scan(scan, matches, frame, dt=CONTAMINATION_CONFIRMATION_SECONDS)
        self.assertEqual(contaminated_items(scan), ["scalpel"])
        self.assertFalse(scan_passed(scan))


class LoadConfigTests(unittest.TestCase):
    """load_config() mutates equipment_logic's module globals directly, so
    every test must restore whatever it touched - otherwise one test's
    calibration override leaks into every test that runs after it."""

    def setUp(self):
        self._originals = {name: getattr(equipment_logic, name) for name in TUNABLE_DEFAULTS}

    def tearDown(self):
        for name, value in self._originals.items():
            setattr(equipment_logic, name, value)

    def test_missing_file_returns_empty_and_leaves_defaults(self):
        applied = load_config(path="/nonexistent/path/config.json")
        self.assertEqual(applied, {})
        self.assertEqual(
            equipment_logic.BLOOD_STAIN_RATIO_THRESHOLD,
            self._originals["BLOOD_STAIN_RATIO_THRESHOLD"],
        )

    def test_applies_a_known_scalar_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"DIRT_STAIN_RATIO_THRESHOLD": 0.5}))
            applied = load_config(path=path)
        self.assertEqual(applied, {"DIRT_STAIN_RATIO_THRESHOLD": 0.5})
        self.assertEqual(equipment_logic.DIRT_STAIN_RATIO_THRESHOLD, 0.5)

    def test_applies_required_equipment_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"REQUIRED_EQUIPMENT": ["scalpel", "towel clamp"]}))
            load_config(path=path)
        self.assertEqual(equipment_logic.REQUIRED_EQUIPMENT, ("scalpel", "towel clamp"))

    def test_ignores_unknown_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"NOT_A_REAL_SETTING": 1}))
            applied = load_config(path=path)
        self.assertEqual(applied, {})

    def test_ignores_malformed_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text("{not valid json")
            applied = load_config(path=path)
        self.assertEqual(applied, {})

    def test_ignores_non_object_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps([1, 2, 3]))
            applied = load_config(path=path)
        self.assertEqual(applied, {})


if __name__ == "__main__":
    unittest.main()
