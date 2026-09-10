"""Pure-drawing HUD for the equipment scan overlay - rounded cards, status
pills, detection-box label chips.

Kept free of the `nanoowl` import, same reasoning as equipment_logic.py:
only cv2/numpy are needed, so the panel's look can be built and actually
inspected - rendered to a PNG, viewed, iterated - without a Jetson,
camera, or NanoOWL engine. nanoowl_equipment_monitor.py is the thin layer
on top that talks to the camera and the real NanoOWL predictor.

Every shape here is a single alpha-blended fill, no per-pixel blur, so
drawing the whole panel stays cheap enough to do every frame on-device -
NanoOWL inference is already this pipeline's FPS ceiling (see
handwash-next/README.md); the HUD shouldn't add to it.
"""

from __future__ import annotations

import cv2
import numpy as np

# =========================================================
# BRAND PALETTE (BGR) - same intent as handwash-next/nanoowl_monitor.py
# ("dark teal-navy card, mint primary, cyan secondary, coral alert") so
# every Argus module reads as one product. PANEL_BG/PANEL_BORDER below
# are NOT copied bit-for-bit from that file: rendering its literal (13,
# 39, 43)/(70, 96, 94) tuples to a PNG and looking at the result showed a
# murky olive-brown, not teal-navy - those tuples read as an RGB triplet
# written directly into a BGR tuple (i.e. red and blue swapped), which
# nothing there could catch since that module needs a Jetson/camera/
# NanoOWL engine just to import. This module doesn't, so the same bug
# would be visible and avoidable here - see equipment-next/README.md.
# =========================================================

PANEL_BG = (43, 39, 13)
PANEL_BORDER = (94, 96, 70)
MINT = (167, 208, 69)
CYAN = (232, 199, 105)
AMBER = (90, 190, 245)
CORAL = (95, 100, 235)
TEXT_BRIGHT = (235, 248, 244)
TEXT_MUTED = (202, 221, 217)
TEXT_FAINT = (140, 168, 165)
DIM = (110, 128, 126)

FONT = cv2.FONT_HERSHEY_SIMPLEX


# =========================================================
# ROUNDED-RECT PRIMITIVES
# =========================================================


def _rounded_mask(height, width, radius):
    """A single-channel mask, 255 inside a `radius`-cornered rounded
    rectangle of the given size, 0 outside."""
    radius = max(0, min(radius, height // 2, width // 2))
    mask = np.zeros((height, width), dtype=np.uint8)
    if radius == 0:
        mask[:] = 255
        return mask
    cv2.rectangle(mask, (radius, 0), (width - radius, height), 255, -1)
    cv2.rectangle(mask, (0, radius), (width, height - radius), 255, -1)
    for cx, cy in (
        (radius, radius),
        (width - radius, radius),
        (radius, height - radius),
        (width - radius, height - radius),
    ):
        cv2.circle(mask, (cx, cy), radius, 255, -1)
    return mask


def draw_rounded_rect(frame, top_left, bottom_right, color, radius=14, alpha=1.0):
    """Alpha-blend a solid rounded rectangle onto `frame` in place.
    Silently clips to the frame and no-ops on a degenerate/off-frame box."""
    x1, x2 = sorted((int(top_left[0]), int(bottom_right[0])))
    y1, y2 = sorted((int(top_left[1]), int(bottom_right[1])))
    x1, x2 = max(x1, 0), min(x2, frame.shape[1])
    y1, y2 = max(y1, 0), min(y2, frame.shape[0])
    height, width = y2 - y1, x2 - x1
    if height <= 0 or width <= 0:
        return

    mask = _rounded_mask(height, width, radius)
    roi = frame[y1:y2, x1:x2]
    fill = np.empty_like(roi)
    fill[:] = color
    blended = cv2.addWeighted(fill, alpha, roi, 1 - alpha, 0)
    mask_bool = mask.astype(bool)
    roi[mask_bool] = blended[mask_bool]


def draw_rounded_border(frame, top_left, bottom_right, color, radius=14, thickness=1):
    x1, x2 = sorted((int(top_left[0]), int(bottom_right[0])))
    y1, y2 = sorted((int(top_left[1]), int(bottom_right[1])))
    x1, x2 = max(x1, 0), min(x2, frame.shape[1])
    y1, y2 = max(y1, 0), min(y2, frame.shape[0])
    height, width = y2 - y1, x2 - x1
    if height <= 0 or width <= 0:
        return

    mask = _rounded_mask(height, width, radius)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(frame[y1:y2, x1:x2], contours, -1, color, thickness, cv2.LINE_AA)


def draw_pill(frame, top_left, bottom_right, color, alpha=1.0):
    """A rounded rect whose radius is half its height - reads as a pill."""
    radius = (int(bottom_right[1]) - int(top_left[1])) // 2
    draw_rounded_rect(frame, top_left, bottom_right, color, radius=radius, alpha=alpha)


# =========================================================
# STATUS GLYPHS
# =========================================================


def _draw_check_glyph(frame, center, color):
    cx, cy = center
    cv2.line(frame, (cx - 4, cy), (cx - 1, cy + 3), color, 2, cv2.LINE_AA)
    cv2.line(frame, (cx - 1, cy + 3), (cx + 4, cy - 3), color, 2, cv2.LINE_AA)


def _draw_alert_glyph(frame, center, color):
    cx, cy = center
    cv2.line(frame, (cx, cy - 4), (cx, cy + 1), color, 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy + 4), 1, color, -1, cv2.LINE_AA)


def draw_status_dot(frame, center, kind):
    """kind: "confirmed" (filled mint + check), "flagged" (filled coral +
    alert), or "pending" (dim ring)."""
    if kind == "confirmed":
        cv2.circle(frame, center, 8, MINT, -1, cv2.LINE_AA)
        _draw_check_glyph(frame, center, PANEL_BG)
    elif kind == "flagged":
        cv2.circle(frame, center, 8, CORAL, -1, cv2.LINE_AA)
        _draw_alert_glyph(frame, center, PANEL_BG)
    else:
        cv2.circle(frame, center, 8, DIM, 2, cv2.LINE_AA)


# =========================================================
# CHECKLIST ITEM PRESENTATION
# =========================================================


def item_kind(item_state):
    """Reduce one equipment_logic scan item to a HUD status: "flagged"
    beats "confirmed" beats "pending" - a contaminated item is never shown
    as merely present."""
    if item_state["blood_flagged"] or item_state["dirt_flagged"]:
        return "flagged"
    if item_state["confirmed"]:
        return "confirmed"
    return "pending"


def item_label(name, item_state):
    """(display name, trailing tag or None) for one checklist row."""
    display = name.title()
    kind = item_kind(item_state)
    if kind == "flagged":
        tag = "BLOODSTAIN?" if item_state["blood_flagged"] else "DIRTY?"
        return display, tag
    if kind == "pending":
        return display, "not seen"
    return display, None


# =========================================================
# DETECTION-BOX LABEL CHIP
# =========================================================


def draw_detection_box(frame, box, label, color):
    """A detection rectangle plus a small filled label chip above it,
    instead of raw stroked text, so labels stay readable over a busy or
    metallic background."""
    x1, y1, x2, y2 = (int(value) for value in box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    (text_w, text_h), _ = cv2.getTextSize(label, FONT, 0.46, 1)
    chip_h = text_h + 10
    chip_top = max(0, y1 - chip_h - 4)
    draw_rounded_rect(
        frame, (x1, chip_top), (x1 + text_w + 14, chip_top + chip_h), color, radius=6, alpha=0.95
    )
    cv2.putText(
        frame, label, (x1 + 7, chip_top + chip_h - 7), FONT, 0.46, PANEL_BG, 1, cv2.LINE_AA
    )


# =========================================================
# CHECKLIST CARD
# =========================================================

CARD_X = 16
CARD_Y = 16
CARD_WIDTH_MAX = 460
CARD_RADIUS = 18
PADDING = 20
HEADER_HEIGHT = 76
ROW_HEIGHT = 32
FOOTER_GAP = 16
PILL_HEIGHT = 36
CONTROLS_GAP = 22
CARD_BOTTOM_PADDING = 16


def card_height_for(item_count):
    return (
        HEADER_HEIGHT
        + item_count * ROW_HEIGHT
        + FOOTER_GAP
        + PILL_HEIGHT
        + CONTROLS_GAP
        + CARD_BOTTOM_PADDING
    )


def _draw_key_chip(frame, origin_x, baseline_y, key, label_text):
    """A small rounded key chip ("R") followed by its label ("Reset").
    Returns the x position immediately after, for chaining chips left to
    right."""
    key_w, key_h = 18, 16
    draw_rounded_rect(
        frame, (origin_x, baseline_y - key_h + 3), (origin_x + key_w, baseline_y + 3),
        PANEL_BORDER, radius=4, alpha=1.0,
    )
    cv2.putText(
        frame, key, (origin_x + 5, baseline_y), FONT, 0.34, TEXT_BRIGHT, 1, cv2.LINE_AA
    )
    label_x = origin_x + key_w + 7
    cv2.putText(frame, label_text, (label_x, baseline_y), FONT, 0.34, TEXT_FAINT, 1, cv2.LINE_AA)
    (label_w, _), _ = cv2.getTextSize(label_text, FONT, 0.34, 1)
    return label_x + label_w + 22


def draw_checklist(frame, scan, ready):
    """Draw the floating equipment-checklist card. `scan` is
    equipment_logic's scan state (a dict of item name -> item state);
    `ready` is the caller's already-computed equipment_logic.scan_passed(scan),
    kept as an explicit argument so this module never has to import
    equipment_logic's pass/fail policy to render it."""

    items = scan["items"]
    width = min(CARD_WIDTH_MAX, frame.shape[1] - CARD_X * 2)
    height = card_height_for(len(items))

    x1, y1 = CARD_X, CARD_Y
    x2, y2 = x1 + width, y1 + height

    # Soft drop shadow (one translucent offset rounded rect, no blur -
    # see module docstring) then the card itself.
    draw_rounded_rect(frame, (x1 + 3, y1 + 6), (x2 + 3, y2 + 6), (0, 0, 0), radius=CARD_RADIUS, alpha=0.30)
    draw_rounded_rect(frame, (x1, y1), (x2, y2), PANEL_BG, radius=CARD_RADIUS, alpha=0.92)
    draw_rounded_border(frame, (x1, y1), (x2, y2), PANEL_BORDER, radius=CARD_RADIUS, thickness=1)

    # ---- header: brand mark + wordmark + subtitle, mode pill ----
    cv2.circle(frame, (x1 + PADDING + 7, y1 + 30), 7, MINT, -1, cv2.LINE_AA)
    cv2.circle(frame, (x1 + PADDING + 16, y1 + 24), 4, CYAN, -1, cv2.LINE_AA)
    cv2.putText(frame, "ARGUS", (x1 + PADDING + 29, y1 + 36), FONT, 0.52, TEXT_BRIGHT, 1, cv2.LINE_AA)
    cv2.putText(
        frame, "Equipment \xb7 General Scan", (x1 + PADDING, y1 + 56),
        FONT, 0.38, TEXT_FAINT, 1, cv2.LINE_AA,
    )

    badge_text = "GENERAL"
    (badge_w, badge_h), _ = cv2.getTextSize(badge_text, FONT, 0.36, 1)
    pill_x2 = x2 - PADDING
    pill_x1 = pill_x2 - badge_w - 22
    pill_y1 = y1 + 18
    pill_y2 = pill_y1 + badge_h + 14
    draw_pill(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), CYAN, alpha=1.0)
    cv2.putText(frame, badge_text, (pill_x1 + 11, pill_y2 - 7), FONT, 0.36, PANEL_BG, 1, cv2.LINE_AA)

    divider_y = y1 + HEADER_HEIGHT - 8
    cv2.line(frame, (x1 + PADDING, divider_y), (x2 - PADDING, divider_y), PANEL_BORDER, 1, cv2.LINE_AA)

    # ---- checklist rows ----
    row_top = y1 + HEADER_HEIGHT
    for index, (name, item_state) in enumerate(items.items()):
        row_center_y = row_top + index * ROW_HEIGHT + ROW_HEIGHT // 2
        kind = item_kind(item_state)
        draw_status_dot(frame, (x1 + PADDING + 8, row_center_y), kind)

        display, tag = item_label(name, item_state)
        name_color = TEXT_MUTED if kind == "pending" else TEXT_BRIGHT
        cv2.putText(
            frame, display, (x1 + PADDING + 26, row_center_y + 5),
            FONT, 0.46, name_color, 1, cv2.LINE_AA,
        )

        if tag:
            tag_color = CORAL if kind == "flagged" else TEXT_FAINT
            (tag_w, _), _ = cv2.getTextSize(tag, FONT, 0.36, 1)
            cv2.putText(
                frame, tag, (x2 - PADDING - tag_w, row_center_y + 5),
                FONT, 0.36, tag_color, 1, cv2.LINE_AA,
            )

    # ---- footer result pill ----
    footer_y1 = row_top + len(items) * ROW_HEIGHT + FOOTER_GAP
    footer_y2 = footer_y1 + PILL_HEIGHT
    footer_x1 = x1 + PADDING
    footer_x2 = x2 - PADDING

    pill_color = MINT if ready else CORAL
    draw_rounded_rect(frame, (footer_x1, footer_y1), (footer_x2, footer_y2), pill_color, radius=PILL_HEIGHT // 2, alpha=0.20)
    draw_rounded_border(frame, (footer_x1, footer_y1), (footer_x2, footer_y2), pill_color, radius=PILL_HEIGHT // 2, thickness=1)

    result_text = "READY FOR USE" if ready else "NOT READY"
    (result_w, result_h), _ = cv2.getTextSize(result_text, FONT, 0.5, 2)
    text_x = footer_x1 + ((footer_x2 - footer_x1) - result_w) // 2
    text_y = footer_y1 + ((footer_y2 - footer_y1) + result_h) // 2 - 2
    cv2.putText(frame, result_text, (text_x, text_y), FONT, 0.5, pill_color, 2, cv2.LINE_AA)

    # ---- controls hint ----
    controls_y = footer_y2 + CONTROLS_GAP
    next_x = _draw_key_chip(frame, x1 + PADDING, controls_y, "R", "Reset")
    _draw_key_chip(frame, next_x, controls_y, "Q", "Quit")
