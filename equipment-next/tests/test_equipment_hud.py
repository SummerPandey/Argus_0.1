import unittest

import numpy as np

from equipment_hud import (
    CORAL,
    DIM,
    MINT,
    draw_checklist,
    draw_detection_box,
    draw_rounded_rect,
    item_kind,
    item_label,
)
from equipment_logic import reset_scan


class RoundedRectTests(unittest.TestCase):
    def test_fills_interior_and_leaves_corner_untouched(self):
        frame = np.zeros((60, 60, 3), dtype=np.uint8)
        draw_rounded_rect(frame, (0, 0), (60, 60), (10, 20, 30), radius=15, alpha=1.0)
        # Interior, away from any corner, is fully filled.
        self.assertTrue((frame[30, 30] == (10, 20, 30)).all())
        # The extreme corner pixel falls outside the rounded mask, so it's
        # untouched - this is what makes the rect actually look rounded
        # rather than square.
        self.assertTrue((frame[0, 0] == (0, 0, 0)).all())

    def test_clips_to_frame_without_raising(self):
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        draw_rounded_rect(frame, (-20, -20), (20, 20), (5, 5, 5), radius=10, alpha=1.0)
        # Center of the visible remainder is well inside the mask; the
        # important thing here is that an off-frame box clips instead of
        # raising an IndexError.
        self.assertTrue((frame[10, 10] == (5, 5, 5)).all())

    def test_degenerate_box_is_a_noop(self):
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        draw_rounded_rect(frame, (10, 10), (10, 10), (5, 5, 5), radius=10, alpha=1.0)
        self.assertTrue((frame == 0).all())

    def test_partial_alpha_blends_rather_than_replaces(self):
        frame = np.full((20, 20, 3), 200, dtype=np.uint8)
        draw_rounded_rect(frame, (0, 0), (20, 20), (0, 0, 0), radius=0, alpha=0.5)
        self.assertTrue((frame[10, 10] == 100).all())


class ItemPresentationTests(unittest.TestCase):
    def test_flagged_beats_confirmed(self):
        item = reset_scan(required=["scalpel"])["items"]["scalpel"]
        item["confirmed"] = True
        item["blood_flagged"] = True
        self.assertEqual(item_kind(item), "flagged")

    def test_confirmed_when_present_and_clean(self):
        item = reset_scan(required=["scalpel"])["items"]["scalpel"]
        item["confirmed"] = True
        self.assertEqual(item_kind(item), "confirmed")

    def test_pending_when_never_seen(self):
        item = reset_scan(required=["scalpel"])["items"]["scalpel"]
        self.assertEqual(item_kind(item), "pending")

    def test_label_tags_match_which_flag_tripped(self):
        item = reset_scan(required=["needle holder"])["items"]["needle holder"]
        item["confirmed"] = True
        item["dirt_flagged"] = True
        display, tag = item_label("needle holder", item)
        self.assertEqual(display, "Needle Holder")
        self.assertEqual(tag, "DIRTY?")


class DrawSmokeTests(unittest.TestCase):
    """These don't assert on exact pixels - just that a realistic call
    doesn't raise and doesn't touch pixels outside the frame, across the
    checklist sizes the general tier actually ships with."""

    def test_draw_checklist_handles_typical_and_edge_item_counts(self):
        for count in (1, 6, 12):
            required = [f"item {i}" for i in range(count)]
            scan = reset_scan(required=required)
            frame = np.zeros((600, 640, 3), dtype=np.uint8)
            draw_checklist(frame, scan, ready=False)
            self.assertEqual(frame.shape, (600, 640, 3))

    def test_draw_checklist_on_a_narrow_frame_does_not_raise(self):
        scan = reset_scan(required=["scalpel", "forceps"])
        frame = np.zeros((400, 200, 3), dtype=np.uint8)
        draw_checklist(frame, scan, ready=True)

    def test_draw_detection_box_near_the_top_edge_does_not_raise(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        draw_detection_box(frame, (10, 0, 50, 20), "SCALPEL", MINT)

    def test_draw_detection_box_colors_are_distinguishable(self):
        self.assertNotEqual(MINT, CORAL)
        self.assertNotEqual(MINT, DIM)


if __name__ == "__main__":
    unittest.main()
