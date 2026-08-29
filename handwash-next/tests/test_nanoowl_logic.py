import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import nanoowl_logic
from nanoowl_logic import (
    REQUIRED_RUB_TIME,
    STALL_HINT_SECONDS,
    TECHNIQUE_STEPS,
    TUNABLE_DEFAULTS,
    WATER_ABSENCE_SECONDS,
    WATER_CONFIRMATION_SECONDS,
    WATER_WET_SECONDS,
    TOWEL_CONFIRMATION_SECONDS,
    FpsMeter,
    advance_dry_evidence,
    advance_faucet_evidence,
    advance_rinse_evidence,
    advance_soap_evidence,
    advance_stall_tracking,
    advance_technique_variation,
    advance_wet_evidence,
    box_near_any,
    boxes_connected,
    detection_name,
    load_config,
    missing_checkpoints,
    motion_ratio,
    procedure_elapsed,
    reset_monitor,
    stall_hint,
    strongest_by_side,
    technique_prompt,
    technique_variation_due,
    union_box,
)


class ReadyMonitor(dict):
    """A monitor dict with every checkpoint satisfied, for checklist tests."""

    def __init__(self):
        super().__init__(reset_monitor())
        self.update(
            wet_confirmed=True,
            soap_seen=True,
            rubbing_time=REQUIRED_RUB_TIME,
            rinse_confirmed=True,
            dry_confirmed=True,
            faucet_confirmed=True,
        )


class ChecklistTests(unittest.TestCase):
    def test_reset_monitor_starts_with_nothing_confirmed(self):
        monitor = reset_monitor()
        self.assertEqual(monitor["state"], "WAITING")
        self.assertFalse(monitor["armed"])
        self.assertIsNone(monitor["result"])

    def test_missing_checkpoints_lists_soap_by_default(self):
        monitor = reset_monitor()
        self.assertIn("soap", missing_checkpoints(monitor))

    def test_missing_checkpoints_room_mode_never_requires_soap(self):
        # Room mode excludes soap from the NanoOWL prompt entirely, so
        # soap_seen can never become True there — it must not be reported
        # as a missing checkpoint.
        monitor = reset_monitor()
        self.assertNotIn("soap", missing_checkpoints(monitor, room_mode=True))

    def test_all_checkpoints_satisfied_leaves_nothing_missing(self):
        self.assertEqual(missing_checkpoints(ReadyMonitor()), [])
        self.assertEqual(missing_checkpoints(ReadyMonitor(), room_mode=True), [])

    def test_room_mode_still_requires_rinse_dry_faucet(self):
        monitor = reset_monitor()
        monitor["wet_confirmed"] = True
        monitor["rubbing_time"] = REQUIRED_RUB_TIME
        missing = missing_checkpoints(monitor, room_mode=True)
        self.assertEqual(missing, ["rinse", "single-use towel", "tap closed with towel"])


class AutomaticWetEvidenceTests(unittest.TestCase):
    def test_confirms_after_three_continuous_seconds(self):
        monitor = reset_monitor()
        # Two seconds of water: not yet enough.
        for _ in range(4):
            confirmed = advance_wet_evidence(monitor, True, dt=0.5)
        self.assertFalse(confirmed)
        self.assertFalse(monitor["wet_confirmed"])
        # One more second crosses WATER_WET_SECONDS (3.0).
        confirmed = advance_wet_evidence(monitor, True, dt=1.0)
        self.assertTrue(confirmed)
        self.assertTrue(monitor["wet_confirmed"])

    def test_interrupted_water_resets_the_timer(self):
        monitor = reset_monitor()
        advance_wet_evidence(monitor, True, dt=2.9)
        advance_wet_evidence(monitor, False, dt=0.1)  # water drops out
        confirmed = advance_wet_evidence(monitor, True, dt=0.2)
        self.assertFalse(confirmed)
        self.assertLess(monitor["water_wet_seconds"], WATER_WET_SECONDS)

    def test_does_nothing_once_rubbing_confirmed(self):
        monitor = reset_monitor()
        monitor["rubbing_confirmed"] = True
        confirmed = advance_wet_evidence(monitor, True, dt=10.0)
        self.assertFalse(confirmed)
        self.assertFalse(monitor["wet_confirmed"])


class AutomaticRinseEvidenceTests(unittest.TestCase):
    def test_ignored_until_rub_target_reached(self):
        monitor = reset_monitor()
        confirmed = advance_rinse_evidence(
            monitor, water_detected=True, rub_target_reached=False, dt=1.0
        )
        self.assertFalse(confirmed)
        self.assertFalse(monitor["rinse_water_seen"])

    def test_confirms_once_water_reappears_then_stops(self):
        monitor = reset_monitor()
        # Water reappears at the tap.
        confirmed = advance_rinse_evidence(
            monitor, water_detected=True, rub_target_reached=True, dt=WATER_CONFIRMATION_SECONDS
        )
        self.assertFalse(confirmed)
        self.assertTrue(monitor["rinse_water_seen"])
        # Water then stops ("stopping the sink").
        confirmed = advance_rinse_evidence(
            monitor, water_detected=False, rub_target_reached=True, dt=WATER_ABSENCE_SECONDS
        )
        self.assertTrue(confirmed)
        self.assertTrue(monitor["rinse_confirmed"])

    def test_flicker_before_confirmation_does_not_confirm(self):
        monitor = reset_monitor()
        advance_rinse_evidence(
            monitor, water_detected=True, rub_target_reached=True, dt=WATER_CONFIRMATION_SECONDS
        )
        # Water blips back on right after disappearing -> gone-timer resets.
        advance_rinse_evidence(monitor, water_detected=False, rub_target_reached=True, dt=0.3)
        advance_rinse_evidence(monitor, water_detected=False, rub_target_reached=True, dt=0.3)
        confirmed = advance_rinse_evidence(
            monitor, water_detected=True, rub_target_reached=True, dt=0.3
        )
        self.assertFalse(confirmed)
        self.assertFalse(monitor["rinse_confirmed"])
        self.assertEqual(monitor["water_gone_seconds"], 0.0)


class AutomaticDryEvidenceTests(unittest.TestCase):
    def test_requires_rinse_first(self):
        monitor = reset_monitor()
        confirmed = advance_dry_evidence(monitor, towel_detected=True, dt=1.0)
        self.assertFalse(confirmed)

    def test_confirms_after_stable_towel_sighting(self):
        monitor = reset_monitor()
        monitor["rinse_confirmed"] = True
        confirmed = advance_dry_evidence(
            monitor, towel_detected=True, dt=TOWEL_CONFIRMATION_SECONDS - 0.1
        )
        self.assertFalse(confirmed)
        confirmed = advance_dry_evidence(monitor, towel_detected=True, dt=0.2)
        self.assertTrue(confirmed)
        self.assertTrue(monitor["dry_confirmed"])


class AutomaticSoapEvidenceTests(unittest.TestCase):
    def test_confirms_after_continuous_seconds(self):
        monitor = reset_monitor()
        confirmed = advance_soap_evidence(monitor, soap_detected=True, dt=0.5)
        self.assertFalse(confirmed)
        confirmed = advance_soap_evidence(monitor, soap_detected=True, dt=0.6)
        self.assertTrue(confirmed)
        self.assertTrue(monitor["soap_seen"])

    def test_interrupted_evidence_resets(self):
        monitor = reset_monitor()
        advance_soap_evidence(monitor, soap_detected=True, dt=0.9)
        advance_soap_evidence(monitor, soap_detected=False, dt=0.1)
        confirmed = advance_soap_evidence(monitor, soap_detected=True, dt=0.2)
        self.assertFalse(confirmed)
        self.assertFalse(monitor["soap_seen"])

    def test_does_nothing_once_already_seen(self):
        monitor = reset_monitor()
        monitor["soap_seen"] = True
        confirmed = advance_soap_evidence(monitor, soap_detected=True, dt=10.0)
        self.assertFalse(confirmed)


class BoxNearAnyTests(unittest.TestCase):
    def test_true_when_close_to_at_least_one_box(self):
        others = [(200, 200, 220, 220), (0, 0, 10, 10)]
        self.assertTrue(box_near_any((5, 5, 15, 15), others, padding=0))

    def test_false_when_far_from_every_box(self):
        others = [(200, 200, 220, 220)]
        self.assertFalse(box_near_any((0, 0, 10, 10), others, padding=5))

    def test_false_for_empty_other_boxes(self):
        self.assertFalse(box_near_any((0, 0, 10, 10), [], padding=1000))


class StallHintTests(unittest.TestCase):
    def test_no_hint_before_threshold(self):
        monitor = reset_monitor()
        monitor["armed"] = True
        advance_stall_tracking(monitor, now=0.0)
        self.assertIsNone(stall_hint(monitor, now=STALL_HINT_SECONDS - 1.0))

    def test_hint_appears_past_threshold_for_wet(self):
        monitor = reset_monitor()
        monitor["armed"] = True
        advance_stall_tracking(monitor, now=0.0)
        self.assertEqual(stall_hint(monitor, now=STALL_HINT_SECONDS), ("wet hands", "W"))

    def test_no_hint_when_nothing_reachable_yet(self):
        # Not armed: wet hasn't started being reachable, and nothing else
        # has a fallback key yet either.
        monitor = reset_monitor()
        advance_stall_tracking(monitor, now=0.0)
        self.assertIsNone(stall_hint(monitor, now=1000.0))

    def test_hint_switches_to_rinse_once_rub_target_reached(self):
        monitor = reset_monitor()
        monitor["armed"] = True
        monitor["wet_confirmed"] = True
        monitor["rubbing_confirmed"] = True
        monitor["rubbing_time"] = REQUIRED_RUB_TIME
        advance_stall_tracking(monitor, now=0.0)
        self.assertEqual(stall_hint(monitor, now=STALL_HINT_SECONDS), ("rinse", "N"))

    def test_timer_resets_when_the_outstanding_step_changes(self):
        monitor = reset_monitor()
        monitor["armed"] = True
        advance_stall_tracking(monitor, now=0.0)
        monitor["wet_confirmed"] = True
        monitor["rubbing_confirmed"] = True
        monitor["rubbing_time"] = REQUIRED_RUB_TIME
        advance_stall_tracking(monitor, now=STALL_HINT_SECONDS)
        # Just switched to waiting on rinse - shouldn't hint immediately.
        self.assertIsNone(stall_hint(monitor, now=STALL_HINT_SECONDS))


class TechniqueVariationTests(unittest.TestCase):
    def test_resets_baseline_on_step_change(self):
        monitor = reset_monitor()
        advance_technique_variation(monitor, rubbing_time=0.0, hand_boxes=[(0, 0, 40, 20)])
        self.assertEqual(monitor["technique_step_index"], 0)
        self.assertFalse(monitor["technique_variation_seen"])

    def test_marks_variation_seen_when_shape_changes_enough(self):
        monitor = reset_monitor()
        advance_technique_variation(monitor, rubbing_time=0.0, hand_boxes=[(0, 0, 40, 20)])
        # Same step, box aspect ratio changes well past the tolerance.
        advance_technique_variation(monitor, rubbing_time=0.1, hand_boxes=[(0, 0, 40, 39)])
        self.assertTrue(monitor["technique_variation_seen"])

    def test_variation_due_only_after_partway_through_the_step(self):
        monitor = reset_monitor()
        advance_technique_variation(monitor, rubbing_time=0.0, hand_boxes=[(0, 0, 40, 20)])
        self.assertFalse(technique_variation_due(monitor, rubbing_time=0.1))
        step_length = REQUIRED_RUB_TIME / len(TECHNIQUE_STEPS)
        self.assertTrue(technique_variation_due(monitor, rubbing_time=step_length * 0.9))

    def test_not_due_once_variation_seen(self):
        monitor = reset_monitor()
        advance_technique_variation(monitor, rubbing_time=0.0, hand_boxes=[(0, 0, 40, 20)])
        advance_technique_variation(monitor, rubbing_time=0.1, hand_boxes=[(0, 0, 40, 39)])
        step_length = REQUIRED_RUB_TIME / len(TECHNIQUE_STEPS)
        self.assertFalse(technique_variation_due(monitor, rubbing_time=step_length * 0.9))


class AutomaticFaucetEvidenceTests(unittest.TestCase):
    def test_requires_dry_first(self):
        monitor = reset_monitor()
        confirmed = advance_faucet_evidence(monitor, towel_faucet_contact=True)
        self.assertFalse(confirmed)

    def test_confirms_on_towel_faucet_contact(self):
        monitor = reset_monitor()
        monitor["dry_confirmed"] = True
        self.assertFalse(advance_faucet_evidence(monitor, towel_faucet_contact=False))
        confirmed = advance_faucet_evidence(monitor, towel_faucet_contact=True)
        self.assertTrue(confirmed)
        self.assertTrue(monitor["faucet_confirmed"])


class TechniquePromptTests(unittest.TestCase):
    def test_first_and_last_step(self):
        self.assertEqual(technique_prompt(0), "Palms together")
        self.assertEqual(technique_prompt(REQUIRED_RUB_TIME), "Rub both fingertips")

    def test_progresses_monotonically(self):
        indices = [
            TECHNIQUE_STEPS.index(technique_prompt(t))
            for t in range(0, int(REQUIRED_RUB_TIME) + 1, 5)
        ]
        self.assertEqual(indices, sorted(indices))
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], len(TECHNIQUE_STEPS) - 1)


class ProcedureElapsedTests(unittest.TestCase):
    def test_none_when_not_started(self):
        self.assertEqual(procedure_elapsed(reset_monitor(), now=100.0), 0.0)

    def test_elapsed_since_start(self):
        monitor = reset_monitor()
        monitor["started_at"] = 10.0
        self.assertEqual(procedure_elapsed(monitor, now=37.5), 27.5)


class GeometryTests(unittest.TestCase):
    def test_overlapping_boxes_are_connected(self):
        self.assertTrue(boxes_connected((0, 0, 10, 10), (5, 5, 15, 15), padding=0))

    def test_boxes_within_padding_are_connected(self):
        self.assertTrue(boxes_connected((0, 0, 10, 10), (20, 0, 30, 10), padding=15))

    def test_boxes_beyond_padding_are_not_connected(self):
        self.assertFalse(boxes_connected((0, 0, 10, 10), (30, 0, 40, 10), padding=5))

    def test_union_box_of_empty_list_is_none(self):
        self.assertIsNone(union_box([], (100, 100)))

    def test_union_box_spans_and_clamps_to_frame(self):
        boxes = [(-10, -10, 20, 20), (80, 80, 200, 200)]
        self.assertEqual(union_box(boxes, (100, 100), padding=0), (0, 0, 100, 100))

    def test_union_box_degenerate_after_clamp_is_none(self):
        # A single point pushed padding=0 still has zero area -> no valid box.
        self.assertIsNone(union_box([(5, 5, 5, 5)], (100, 100)))


class MotionRatioTests(unittest.TestCase):
    def test_no_previous_frame_returns_zero(self):
        frame = np.zeros((50, 50), dtype=np.uint8)
        self.assertEqual(motion_ratio(frame, None, (0, 0, 50, 50)), 0.0)

    def test_identical_frames_have_no_motion(self):
        frame = np.full((50, 50), 128, dtype=np.uint8)
        self.assertEqual(motion_ratio(frame, frame.copy(), (0, 0, 50, 50)), 0.0)

    def test_changed_region_registers_motion(self):
        previous = np.zeros((50, 50), dtype=np.uint8)
        current = previous.copy()
        current[10:40, 10:40] = 255
        ratio = motion_ratio(current, previous, (0, 0, 50, 50))
        self.assertGreater(ratio, 0.3)


class StrongestBySideTests(unittest.TestCase):
    def test_picks_by_explicit_left_right_label(self):
        items = [
            {"label": "a right hand", "score": 0.9, "box": (60, 0, 80, 20)},
            {"label": "a left hand", "score": 0.4, "box": (0, 0, 20, 20)},
            {"label": "a left hand", "score": 0.8, "box": (5, 5, 25, 25)},
        ]
        left, right = strongest_by_side(items)
        self.assertEqual(left["score"], 0.8)
        self.assertEqual(right["score"], 0.9)

    def test_falls_back_to_position_when_labels_missing(self):
        items = [
            {"label": "a hand", "score": 0.9, "box": (60, 0, 80, 20)},
            {"label": "a hand", "score": 0.7, "box": (0, 0, 20, 20)},
        ]
        left, right = strongest_by_side(items)
        self.assertEqual(left["box"][0], 0)
        self.assertEqual(right["box"][0], 60)


class DetectionNameTests(unittest.TestCase):
    def test_resolves_label_from_tree(self):
        detection = SimpleNamespace(labels=[2])
        tree = SimpleNamespace(labels=["a left hand", "a right hand", "soap foam"])
        self.assertEqual(detection_name(detection, tree), "soap foam")

    def test_returns_unknown_when_out_of_range_or_empty(self):
        tree = SimpleNamespace(labels=["a left hand"])
        self.assertEqual(detection_name(SimpleNamespace(labels=[]), tree), "unknown")
        self.assertEqual(detection_name(SimpleNamespace(labels=[9]), tree), "unknown")


class LoadConfigTests(unittest.TestCase):
    """load_config() mutates nanoowl_logic's module globals directly, so
    every test must restore whatever it touched - otherwise one test's
    calibration override leaks into every test that runs after it."""

    def setUp(self):
        self._originals = {name: getattr(nanoowl_logic, name) for name in TUNABLE_DEFAULTS}

    def tearDown(self):
        for name, value in self._originals.items():
            setattr(nanoowl_logic, name, value)

    def test_missing_file_returns_empty_and_leaves_defaults(self):
        applied = load_config(path="/nonexistent/path/config.json")
        self.assertEqual(applied, {})
        self.assertEqual(
            nanoowl_logic.WATER_EVIDENCE_THRESHOLD,
            self._originals["WATER_EVIDENCE_THRESHOLD"],
        )

    def test_applies_a_known_key_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"WATER_EVIDENCE_THRESHOLD": 0.42}))
            applied = load_config(path=path)
        self.assertEqual(applied, {"WATER_EVIDENCE_THRESHOLD": 0.42})
        self.assertEqual(nanoowl_logic.WATER_EVIDENCE_THRESHOLD, 0.42)

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


class FpsMeterTests(unittest.TestCase):
    """Timestamps are caller-supplied, so every case here is exact."""

    def test_no_ticks_reads_zero(self):
        meter = FpsMeter()
        self.assertEqual(meter.rate(5.0), 0.0)

    def test_steady_rate_is_measured(self):
        meter = FpsMeter(window_seconds=2.0)
        now = 100.0
        for index in range(120):  # 60 FPS for two full seconds
            now = 100.0 + index / 60.0
            meter.tick(now)
        self.assertAlmostEqual(meter.rate(now), 60.0, delta=1.5)

    def test_reads_correctly_during_the_first_partial_window(self):
        # Startup must not under-report by dividing a half-second of
        # samples by the full window.
        meter = FpsMeter(window_seconds=2.0)
        now = 0.0
        for index in range(15):  # 30 FPS for half a second
            now = index / 30.0
            meter.tick(now)
        self.assertAlmostEqual(meter.rate(now), 30.0, delta=3.0)

    def test_rate_decays_toward_zero_once_ticks_stop(self):
        # The paused-inference case: during a result screen no frames are
        # submitted, and the HUD must fall to 0 rather than freeze on the
        # last healthy number.
        meter = FpsMeter(window_seconds=2.0)
        for index in range(60):
            meter.tick(index / 30.0)
        busy = meter.rate(2.0)
        self.assertGreater(busy, 20.0)
        self.assertLess(meter.rate(3.0), busy)
        self.assertEqual(meter.rate(6.0), 0.0)

    def test_reset_clears_history(self):
        meter = FpsMeter()
        for index in range(30):
            meter.tick(index / 30.0)
        meter.reset()
        self.assertEqual(meter.rate(1.0), 0.0)


if __name__ == "__main__":
    unittest.main()
