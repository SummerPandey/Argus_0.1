"""Detection-agnostic checklist and contamination logic for the "general"
equipment scan.

Deliberately free of the `nanoowl`/TensorRT imports so it can be unit
tested on any machine (no Jetson, no camera, no engine file required),
the same way handwash-next/nanoowl_logic.py is. nanoowl_equipment_monitor.py
is the only piece of this module that needs the real NanoOWL container.

Scope, on purpose: this is the "general" tier only - a configurable
baseline checklist ("is the bare-minimum tray present") plus a visual
contamination heuristic ("does anything look bloodstained or dirty").
It is not a sterility check, does not identify specific instrument
models, and is not a certified clinical device - see the module README.

By design, and matching the rest of Argus, nothing in this module writes
scan results anywhere. Results are shown live on screen and then discarded
- no CSV, no database, no network call. The only file this module ever
touches is an optional local config.json read at startup for calibration
(see load_config below); it never writes one.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

# =========================================================
# CHECKLIST: BARE-MINIMUM GENERAL TRAY
# =========================================================

# A reasonable starting checklist for a general-purpose instrument tray,
# not a clinically validated or facility-specific requirement. Edit via
# config.json's REQUIRED_EQUIPMENT (a list of strings) to match an actual
# tray instead of changing this in place - see TUNABLE_DEFAULTS below.
DEFAULT_REQUIRED_EQUIPMENT = (
    "scalpel",
    "forceps",
    "scissors",
    "hemostat",
    "needle holder",
    "retractor",
)
REQUIRED_EQUIPMENT = DEFAULT_REQUIRED_EQUIPMENT

# NanoOWL's own per-box acceptance cutoff for the equipment prompt.
DETECTION_THRESHOLD = 0.10

# An item must be seen continuously this long before it counts as
# "present" (debounces a single lucky/unlucky frame).
ITEM_CONFIRMATION_SECONDS = 1.0

# Once confirmed present, a brief miss (occlusion, a hand passing over the
# tray, one bad frame) is tolerated for this long before the item flips
# back to "missing" - mirrors handwash's MAX_SEPARATION_TIME grace period.
ITEM_ABSENCE_GRACE_SECONDS = 1.5

# =========================================================
# CONTAMINATION HEURISTIC
# =========================================================
# Box detectors like NanoOWL give a location, not a material judgement,
# so "contaminated" here is a classical-CV color heuristic over the pixels
# inside an item's own detection box, not a verified stain/pathogen check.
# All ratios/HSV bands below are uncalibrated starting points (same
# caveat nanoowl_logic.py makes for its own thresholds) - tune them on
# real footage of the actual tray, lighting, and camera via config.json
# rather than editing this file.

# Fraction of an item's crop that must read as blood-red before the
# blood flag trips.
BLOOD_STAIN_RATIO_THRESHOLD = 0.03

# Fraction of an item's crop that must read as brown/dirt-toned before
# the dirt flag trips.
DIRT_STAIN_RATIO_THRESHOLD = 0.08

# A flagged contamination reading must persist this long before it
# latches - see advance_item's docstring for why it then stays latched.
CONTAMINATION_CONFIRMATION_SECONDS = 1.0

# HSV bands, OpenCV convention (H: 0-179, S/V: 0-255). Blood - fresh or
# dried - reads as a red hue with a value/saturation floor that excludes
# both near-black shadow and near-white glare off polished steel; red
# wraps around hue 0, hence two ranges.
BLOOD_HUE_RANGES = ((0, 10), (165, 179))
BLOOD_MIN_SATURATION = 80
BLOOD_VALUE_RANGE = (20, 200)

# Dirt/grime - a brown/yellow hue, darker and less saturated than the
# bright neutral gray of clean stainless steel.
DIRT_HUE_RANGE = (10, 35)
DIRT_MIN_SATURATION = 40
DIRT_VALUE_RANGE = (20, 150)


# =========================================================
# IMAGE HELPERS
# =========================================================


def crop_box(frame, box):
    """Return the sub-image `box` covers, clamped to the frame bounds, or
    None if the clamped box has no area (fully off-frame, degenerate)."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(value) for value in box)
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _hsv_ratio(crop_bgr, hue_ranges, min_saturation, value_range):
    """Fraction of pixels in `crop_bgr` (BGR, uint8) whose hue falls in any
    of `hue_ranges` and whose saturation/value fall within the given
    floors/range. Returns 0.0 for an empty or missing crop."""
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    hue_mask = np.zeros(hue.shape, dtype=bool)
    for low, high in hue_ranges:
        hue_mask |= (hue >= low) & (hue <= high)

    mask = (
        hue_mask
        & (saturation >= min_saturation)
        & (value >= value_range[0])
        & (value <= value_range[1])
    )
    return float(np.count_nonzero(mask)) / mask.size


def blood_stain_ratio(crop_bgr):
    return _hsv_ratio(crop_bgr, BLOOD_HUE_RANGES, BLOOD_MIN_SATURATION, BLOOD_VALUE_RANGE)


def dirt_stain_ratio(crop_bgr):
    return _hsv_ratio(crop_bgr, (DIRT_HUE_RANGE,), DIRT_MIN_SATURATION, DIRT_VALUE_RANGE)


# =========================================================
# CHECKLIST MATCHING
# =========================================================


def matches_required(label, required_name):
    """Loose match between a NanoOWL label (e.g. "a needle holder") and a
    required checklist name (e.g. "needle holder") - substring either
    direction, case-insensitive, so prompt phrasing doesn't have to match
    the checklist name exactly."""
    label = label.lower()
    required_name = required_name.lower()
    return required_name in label or label in required_name


def best_match_for(required_name, detections):
    """Strongest detection matching `required_name`, or None if none do."""
    candidates = [item for item in detections if matches_required(item["label"], required_name)]
    return max(candidates, key=lambda item: item["score"], default=None)


def match_checklist(detections, required=None):
    """Map each required item name to its best current-frame detection (or
    None). `required` defaults to the live REQUIRED_EQUIPMENT global so a
    config.json override applies without needing to re-import."""
    names = REQUIRED_EQUIPMENT if required is None else required
    return {name: best_match_for(name, detections) for name in names}


# =========================================================
# SCAN STATE
# =========================================================


def reset_scan(required=None):
    """A fresh scan: nothing confirmed present, nothing flagged."""
    names = REQUIRED_EQUIPMENT if required is None else required
    return {
        "required": tuple(names),
        "items": {
            name: {
                "present_seconds": 0.0,
                "confirmed": False,
                "absent_seconds": 0.0,
                "blood_seconds": 0.0,
                "dirt_seconds": 0.0,
                "blood_flagged": False,
                "dirt_flagged": False,
                "last_box": None,
                "last_score": 0.0,
            }
            for name in names
        },
    }


def advance_item(item_state, detection, frame, dt):
    """Update one checklist item's presence/contamination timers for a
    single frame. `detection` is the best current-frame match for this
    item (or None if not seen this frame); `frame` is the full BGR frame
    the detection's box was found in, for cropping the contamination
    check. Mutates and returns item_state.

    Contamination flags are deliberately sticky: once latched, they stay
    latched for the rest of this scan (cleared only by reset_scan), rather
    than clearing the moment a frame happens to look clean. A glare angle
    or a hand passing over the item could otherwise hide an already-seen
    stain and silently clear a real flag - safer to require a human to
    reset after re-checking/cleaning the item than to risk that.
    """
    if detection is None:
        item_state["present_seconds"] = 0.0
        if item_state["confirmed"]:
            item_state["absent_seconds"] += dt
            if item_state["absent_seconds"] >= ITEM_ABSENCE_GRACE_SECONDS:
                item_state["confirmed"] = False
        return item_state

    item_state["last_box"] = detection["box"]
    item_state["last_score"] = detection["score"]
    item_state["absent_seconds"] = 0.0
    item_state["present_seconds"] += dt
    if item_state["present_seconds"] >= ITEM_CONFIRMATION_SECONDS:
        item_state["confirmed"] = True

    crop = crop_box(frame, detection["box"])
    blood_ratio = blood_stain_ratio(crop)
    dirt_ratio = dirt_stain_ratio(crop)

    item_state["blood_seconds"] = (
        item_state["blood_seconds"] + dt
        if blood_ratio >= BLOOD_STAIN_RATIO_THRESHOLD
        else 0.0
    )
    item_state["dirt_seconds"] = (
        item_state["dirt_seconds"] + dt
        if dirt_ratio >= DIRT_STAIN_RATIO_THRESHOLD
        else 0.0
    )

    if item_state["blood_seconds"] >= CONTAMINATION_CONFIRMATION_SECONDS:
        item_state["blood_flagged"] = True
    if item_state["dirt_seconds"] >= CONTAMINATION_CONFIRMATION_SECONDS:
        item_state["dirt_flagged"] = True

    return item_state


def advance_scan(scan, matches, frame, dt):
    """Advance every checklist item one frame. `matches` is
    match_checklist()'s output for this frame."""
    for name, item_state in scan["items"].items():
        advance_item(item_state, matches.get(name), frame, dt)
    return scan


# =========================================================
# RESULTS
# =========================================================


def missing_items(scan):
    return [name for name, item in scan["items"].items() if not item["confirmed"]]


def contaminated_items(scan):
    return [
        name
        for name, item in scan["items"].items()
        if item["blood_flagged"] or item["dirt_flagged"]
    ]


def scan_passed(scan):
    """True once every required item is confirmed present and nothing has
    latched a contamination flag."""
    return not missing_items(scan) and not contaminated_items(scan)


# =========================================================
# ON-DEVICE CALIBRATION
# =========================================================
# Same mechanism as handwash-next/nanoowl_logic.py's load_config: every
# value above that's plausibly worth tuning per-camera/per-tray, captured
# as {name: default} so load_config() can validate a config.json against
# it. Calibration only - reads settings in, never writes anything.
TUNABLE_DEFAULTS = {
    "REQUIRED_EQUIPMENT": list(REQUIRED_EQUIPMENT),
    "DETECTION_THRESHOLD": DETECTION_THRESHOLD,
    "ITEM_CONFIRMATION_SECONDS": ITEM_CONFIRMATION_SECONDS,
    "ITEM_ABSENCE_GRACE_SECONDS": ITEM_ABSENCE_GRACE_SECONDS,
    "BLOOD_STAIN_RATIO_THRESHOLD": BLOOD_STAIN_RATIO_THRESHOLD,
    "DIRT_STAIN_RATIO_THRESHOLD": DIRT_STAIN_RATIO_THRESHOLD,
    "CONTAMINATION_CONFIRMATION_SECONDS": CONTAMINATION_CONFIRMATION_SECONDS,
    "BLOOD_MIN_SATURATION": BLOOD_MIN_SATURATION,
    "DIRT_MIN_SATURATION": DIRT_MIN_SATURATION,
}

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def load_config(path=None):
    """Apply on-device calibration overrides from a local JSON file.

    Reads {"REQUIRED_EQUIPMENT": [...], "BLOOD_STAIN_RATIO_THRESHOLD": 0.05,
    ...} and overwrites the matching module-level defaults above, so
    nanoowl_equipment_monitor.py (and this module's own functions) pick up
    the tuned values. Missing file, empty file, or unknown keys are all
    handled quietly rather than crashing the app over a calibration typo -
    this is the one file this module ever reads, and it never writes one.

    Must run before nanoowl_equipment_monitor.py does
    `from equipment_logic import THRESHOLD_NAME`, since that copies the
    value at that moment.
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}

    try:
        with path.open() as stream:
            overrides = json.load(stream)
    except (OSError, ValueError) as exc:
        print(f"Ignoring unreadable config at {path}: {exc}")
        return {}

    if not isinstance(overrides, dict):
        print(f"Ignoring config at {path}: expected a JSON object.")
        return {}

    applied = {}
    for name, value in overrides.items():
        if name not in TUNABLE_DEFAULTS:
            print(f"Ignoring unknown config key: {name}")
            continue
        if name == "REQUIRED_EQUIPMENT":
            value = tuple(value)
        globals()[name] = value
        applied[name] = value
    return applied
