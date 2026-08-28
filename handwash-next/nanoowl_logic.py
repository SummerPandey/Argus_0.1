"""Detection-agnostic geometry, timing, and checklist logic for the NanoOWL
sink-test monitor.

Deliberately free of the `nanoowl`/TensorRT imports so it can be unit tested
on any machine (no Jetson, no camera, no engine file required) even though
nanoowl_monitor.py itself only ever runs inside the NanoOWL container.

By design, nothing in this module writes session, detection, or identity
data anywhere. Results are shown live on screen and then discarded - no
CSV, no database, no network call. The only file this module ever touches
is an optional local config.json read at startup for threshold calibration
(see load_config below); it never writes one.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

# =========================================================
# HANDWASH RULES
# =========================================================

# WHO soap-and-water handwashing takes 40-60 seconds for the entire procedure.
# NanoOWL measures active rubbing separately; the remaining time covers wetting,
# rinsing, drying, and closing the tap with the towel.
REQUIRED_RUB_TIME = 30.0
MINIMUM_WASH_TIME = 40.0
MAXIMUM_WASH_TIME = 60.0

# Hands must initially remain together/rubbing for five continuous seconds.
INITIAL_CONFIRMATION_TIME = 5.0

# Once washing has been confirmed, hands may be separated briefly without
# losing the accumulated evidence. Ten full seconds apart marks the rubbing
# attempt incomplete.
MAX_SEPARATION_TIME = 10.0

# Red/green screen remains for 5 seconds, then the entire session resets.
RESULT_DISPLAY_TIME = 5.0

# Guided WHO surface-coverage sequence. NanoOWL's boxes cannot prove the hand
# pose, so these are coaching prompts and are logged as guided, not verified.
TECHNIQUE_STEPS = (
    "Palms together",
    "Backs of hands",
    "Between fingers",
    "Backs of fingers",
    "Rotate both thumbs",
    "Rub both fingertips",
)


# =========================================================
# DETECTION GEOMETRY
# =========================================================

# Small allowed gap between two hand boxes.
BOX_CONNECTION_PADDING = 35

# Allowed gap between forearms.
MAX_FOREARM_GAP = 120

# Padding around hand/forearm motion region.
ROI_PADDING = 25

# Allowed gap between a towel box and a faucet box to count as contact.
TOWEL_FAUCET_PADDING = 40


# =========================================================
# MOTION SETTINGS
# =========================================================

MOTION_PIXEL_THRESHOLD = 20
MINIMUM_MOTION_RATIO = 0.025


# =========================================================
# AUTOMATIC CHECKPOINT EVIDENCE
# =========================================================

# These thresholds drive automatic detection of the steps a plain
# hand/forearm/soap detector can't otherwise verify (wet, rinse, dry, tap
# closed with towel), replacing manual W/N/D/F key presses for the real
# sink test. nanoowl_monitor.py keeps the keys wired as a manual fallback
# in case detection misses on-device, since open-vocabulary detectors are
# generally weaker at amorphous/transparent things like running water than
# at solid objects.
#
# All confirmation windows below are wall-clock seconds, not frame counts
# - NanoOWL's inference latency isn't constant (thermal throttling, engine
# warmup), so a frame-count debounce would silently stretch or shrink in
# real time. Accumulating dt keeps the feel consistent regardless of FPS.
WATER_WET_SECONDS = 3.0  # continuous water -> WHO step 1 (wet hands)
WATER_CONFIRMATION_SECONDS = 0.8  # water must reappear continuously this long
WATER_ABSENCE_SECONDS = 0.8  # water must then stop continuously this long
TOWEL_CONFIRMATION_SECONDS = 0.8  # a towel must be seen continuously this long
SOAP_CONFIRMATION_SECONDS = 1.0  # soap/foam must be seen continuously this long

# Water/soap evidence only counts toward the wet/rinse/soap checkpoints
# when it's actually near the hands (or forearms) currently in frame, so a
# tap left running with no hands under it can't fake a checkpoint. The
# caller skips this requirement when no hand/forearm box is visible at
# all, so momentary hand occlusion doesn't itself break detection.
HAND_PROXIMITY_PADDING = 160

# How long a checkpoint that has a manual W/N/D/F fallback can sit
# outstanding before the UI nudges the operator toward pressing it - see
# advance_stall_tracking/stall_hint below.
STALL_HINT_SECONDS = 8.0

# Coaching-only nudge (see advance_technique_variation): how much the
# union-of-hands box aspect ratio must change to count as "the grip
# changed" during a single WHO technique step.
GRIP_VARIATION_TOLERANCE = 0.12


# =========================================================
# CAMERA + DETECTION THRESHOLDS
# =========================================================

# Everything below is a starting-point guess, not calibrated against real
# footage - tune it on-device (see load_config at the bottom of this file)
# rather than editing these in place.
CAMERA_ID = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# How long camera.read() can keep failing (dropped USB frame, driver
# hiccup) before the app gives up and exits, instead of tearing down the
# whole session on one bad read. Reconnect attempts happen in between.
CAMERA_RETRY_SECONDS = 5.0

DETECTION_THRESHOLD = 0.10  # NanoOWL's own per-box acceptance cutoff
SOAP_EVIDENCE_THRESHOLD = 0.18
WATER_EVIDENCE_THRESHOLD = 0.15
TOWEL_EVIDENCE_THRESHOLD = 0.15

# Ceiling on how often frames are handed to the NanoOWL worker. The
# checklist logic is dt-accumulated wall-clock time, so it's cadence-
# independent; anything past ~15 checks per second buys no responsiveness
# and just heats the Jetson toward thermal throttling, which slows the
# sustained rate. Set higher (or lower) per-device via config.json. Room
# hand test applies the same idea to MediaPipe (bubbles/camera.py's
# max_inference_fps).
MAX_INFERENCE_FPS = 15.0


# =========================================================
# GEOMETRY HELPERS
# =========================================================


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def boxes_connected(box1, box2, padding):
    a_x1, a_y1, a_x2, a_y2 = box1
    b_x1, b_y1, b_x2, b_y2 = box2
    return (
        a_x1 <= b_x2 + padding
        and b_x1 <= a_x2 + padding
        and a_y1 <= b_y2 + padding
        and b_y1 <= a_y2 + padding
    )


def box_near_any(box, other_boxes, padding):
    """True if `box` is within `padding` of at least one box in
    `other_boxes`. Used to require water/soap evidence be near the hands
    actually in frame, not just present anywhere in the picture."""
    return any(boxes_connected(box, other, padding) for other in other_boxes)


def union_box(boxes, frame_shape, padding=0):
    if not boxes:
        return None

    height, width = frame_shape[:2]
    x1 = max(0, int(min(box[0] for box in boxes) - padding))
    y1 = max(0, int(min(box[1] for box in boxes) - padding))
    x2 = min(width, int(max(box[2] for box in boxes) + padding))
    y2 = min(height, int(max(box[3] for box in boxes) + padding))

    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


# =========================================================
# NANOOWL LABEL HELPERS
# =========================================================


def detection_name(detection, tree):
    if len(detection.labels) == 0:
        return "unknown"
    label_index = int(detection.labels[-1])
    if 0 <= label_index < len(tree.labels):
        return str(tree.labels[label_index])
    return "unknown"


def strongest_by_side(items):
    left_items = [item for item in items if "left" in item["label"].lower()]
    right_items = [item for item in items if "right" in item["label"].lower()]

    left = max(left_items, key=lambda item: item["score"], default=None)
    right = max(right_items, key=lambda item: item["score"], default=None)

    # Fallback: if NanoOWL does not consistently label left/right, use the
    # two strongest boxes and determine sides from screen position.
    if (left is None or right is None) and len(items) >= 2:
        selected = sorted(items, key=lambda item: item["score"], reverse=True)[:2]
        selected.sort(key=lambda item: box_center(item["box"])[0])
        left, right = selected

    return left, right


# =========================================================
# MOTION DETECTION
# =========================================================


def motion_ratio(current_gray, previous_gray, roi):
    if previous_gray is None or roi is None:
        return 0.0

    x1, y1, x2, y2 = roi
    current_roi = current_gray[y1:y2, x1:x2]
    previous_roi = previous_gray[y1:y2, x1:x2]

    if current_roi.size == 0:
        return 0.0
    if previous_roi.shape != current_roi.shape:
        return 0.0

    current_roi = cv2.GaussianBlur(current_roi, (7, 7), 0)
    previous_roi = cv2.GaussianBlur(previous_roi, (7, 7), 0)
    difference = cv2.absdiff(current_roi, previous_roi)
    changed = difference >= MOTION_PIXEL_THRESHOLD
    return float(np.count_nonzero(changed)) / changed.size


# =========================================================
# PROCEDURE / CHECKLIST STATE
# =========================================================


def _technique_step_length():
    return REQUIRED_RUB_TIME / len(TECHNIQUE_STEPS)


def technique_step_index(rubbing_time):
    return min(
        len(TECHNIQUE_STEPS) - 1, int(rubbing_time / _technique_step_length())
    )


def technique_step_elapsed(rubbing_time):
    """Seconds spent within the *current* coached step, for pacing the
    grip-variation nudge below."""
    return rubbing_time % _technique_step_length()


def technique_prompt(rubbing_time):
    """Return the current coached WHO coverage step (not a verified pose)."""
    return TECHNIQUE_STEPS[technique_step_index(rubbing_time)]


def hand_grip_signature(hand_boxes):
    """Rough proxy for hand configuration from box geometry alone - NanoOWL
    gives boxes, not pose, so this is the width/height ratio of the union
    of visible hand boxes. Different WHO grips (palms flat, interlaced
    fingers, thumb rotation) tend to shift this ratio even though the
    actual pose can't be seen directly. Coaching signal only."""
    box = union_box(hand_boxes, (100_000, 100_000))
    if box is None:
        return None
    x1, y1, x2, y2 = box
    height = y2 - y1
    return (x2 - x1) / height if height else None


def advance_technique_variation(monitor, rubbing_time, hand_boxes):
    """Track whether hand geometry has visibly changed since the current
    coached technique step began, so the UI can nudge someone who's just
    holding one fixed grip through all six steps. Coaching-only - the
    steps themselves stay time-paced (technique_prompt) since box-only
    detection can't verify pose, so this never blocks progress."""
    step = technique_step_index(rubbing_time)
    signature = hand_grip_signature(hand_boxes)

    if step != monitor["technique_step_index"]:
        monitor["technique_step_index"] = step
        monitor["technique_signature"] = signature
        monitor["technique_variation_seen"] = False
        return

    if signature is None or monitor["technique_signature"] is None:
        return

    if abs(signature - monitor["technique_signature"]) >= GRIP_VARIATION_TOLERANCE:
        monitor["technique_variation_seen"] = True


def technique_variation_due(monitor, rubbing_time):
    """True once we're far enough into the current coached step that a
    still-unchanged grip is worth a gentle on-screen nudge."""
    if monitor["technique_variation_seen"]:
        return False
    return technique_step_elapsed(rubbing_time) >= _technique_step_length() * 0.6


def procedure_elapsed(monitor, now):
    started = monitor.get("started_at")
    return 0.0 if started is None else max(0.0, now - started)


def missing_checkpoints(monitor, room_mode=False):
    """Checkpoints still required before the session can complete.

    Room mode deliberately never observes soap (it's excluded from the
    NanoOWL prompt, so soap_seen can never become True), so it must be
    skipped here rather than reported as a missing step.
    """
    checks = [
        ("wet hands", monitor["wet_confirmed"]),
        ("30s active rubbing", monitor["rubbing_time"] >= REQUIRED_RUB_TIME),
        ("rinse", monitor["rinse_confirmed"]),
        ("single-use towel", monitor["dry_confirmed"]),
        ("tap closed with towel", monitor["faucet_confirmed"]),
    ]
    if not room_mode:
        checks.insert(1, ("soap", monitor["soap_seen"]))
    return [name for name, complete in checks if not complete]


def advance_wet_evidence(monitor, water_detected, dt):
    """WHO step 1 (wet hands): auto-confirm once water has been seen
    continuously for WATER_WET_SECONDS. Returns True the frame this
    becomes newly confirmed, so the caller can start the WHO clock."""
    if monitor["wet_confirmed"] or monitor["rubbing_confirmed"]:
        return False
    monitor["water_wet_seconds"] = (
        monitor["water_wet_seconds"] + dt if water_detected else 0.0
    )
    if monitor["water_wet_seconds"] >= WATER_WET_SECONDS:
        monitor["wet_confirmed"] = True
        return True
    return False


def advance_rinse_evidence(monitor, water_detected, rub_target_reached, dt):
    """WHO step 9 (rinse): once active rubbing is done, auto-confirm once
    water reappears at the tap and then stops again — "stopping the sink"
    after rinsing. Returns True the frame this becomes newly confirmed."""
    if monitor["rinse_confirmed"] or not rub_target_reached:
        return False

    if not monitor["rinse_water_seen"]:
        monitor["water_seen_seconds"] = (
            monitor["water_seen_seconds"] + dt if water_detected else 0.0
        )
        if monitor["water_seen_seconds"] >= WATER_CONFIRMATION_SECONDS:
            monitor["rinse_water_seen"] = True
            monitor["water_gone_seconds"] = 0.0
        return False

    monitor["water_gone_seconds"] = (
        0.0 if water_detected else monitor["water_gone_seconds"] + dt
    )
    if monitor["water_gone_seconds"] >= WATER_ABSENCE_SECONDS:
        monitor["rinse_confirmed"] = True
        return True
    return False


def advance_dry_evidence(monitor, towel_detected, dt):
    """WHO step 10 (dry): once rinsed, auto-confirm 'dried' as soon as a
    towel is reliably seen. Returns True the frame this becomes newly
    confirmed."""
    if monitor["dry_confirmed"] or not monitor["rinse_confirmed"]:
        return False
    monitor["towel_seconds"] = (
        monitor["towel_seconds"] + dt if towel_detected else 0.0
    )
    if monitor["towel_seconds"] >= TOWEL_CONFIRMATION_SECONDS:
        monitor["dry_confirmed"] = True
        return True
    return False


def advance_faucet_evidence(monitor, towel_faucet_contact):
    """WHO step 11 (tap off with towel): once dried, auto-confirm as soon
    as the towel box touches the faucet box. Returns True the frame this
    becomes newly confirmed."""
    if monitor["faucet_confirmed"] or not monitor["dry_confirmed"]:
        return False
    if towel_faucet_contact:
        monitor["faucet_confirmed"] = True
        return True
    return False


def advance_soap_evidence(monitor, soap_detected, dt):
    """WHO step 2 (soap): auto-confirm once soap/foam evidence has been
    seen continuously for SOAP_CONFIRMATION_SECONDS. `soap_detected`
    should already reflect the score threshold, armed/wet-confirmed
    gating, and hand-proximity check - this function only debounces it."""
    if monitor["soap_seen"]:
        return False
    monitor["soap_evidence_seconds"] = (
        monitor["soap_evidence_seconds"] + dt if soap_detected else 0.0
    )
    if monitor["soap_evidence_seconds"] >= SOAP_CONFIRMATION_SECONDS:
        monitor["soap_seen"] = True
        return True
    return False


# =========================================================
# STALL HINTS
# =========================================================
# The auto-detected checkpoints (wet/rinse/dry/faucet) each have a manual
# W/N/D/F fallback key for when on-device detection misses, but nothing
# told the operator that fallback existed or that it was time to use it.
# This tracks how long the *next reachable* checkpoint has been
# outstanding and, past STALL_HINT_SECONDS, tells the UI which key to
# suggest. The gating below intentionally mirrors nanoowl_monitor.py's
# key-press conditions exactly, so a hint is never shown for a key that
# wouldn't actually do anything yet.


def _stall_candidates(monitor):
    yield "wet hands", "W", (
        monitor["armed"]
        and not monitor["rubbing_confirmed"]
        and not monitor["wet_confirmed"]
    )
    yield "rinse", "N", (
        monitor["rubbing_time"] >= REQUIRED_RUB_TIME and not monitor["rinse_confirmed"]
    )
    yield "single-use towel", "D", (
        monitor["rinse_confirmed"] and not monitor["dry_confirmed"]
    )
    yield "tap closed with towel", "F", (
        monitor["dry_confirmed"] and not monitor["faucet_confirmed"]
    )


def advance_stall_tracking(monitor, now):
    """Call once per frame. Records when the currently-outstanding
    fallback-eligible checkpoint changed, so stall_hint can measure how
    long it's been stuck."""
    step = next((name for name, _key, gate in _stall_candidates(monitor) if gate), None)
    if step != monitor["waiting_step"]:
        monitor["waiting_step"] = step
        monitor["waiting_since"] = now if step else None


def stall_hint(monitor, now):
    """Return (checkpoint_name, fallback_key) once the outstanding
    checkpoint has sat unconfirmed for STALL_HINT_SECONDS, else None."""
    step = monitor["waiting_step"]
    since = monitor["waiting_since"]
    if step is None or since is None or now - since < STALL_HINT_SECONDS:
        return None
    return next(
        (name, key) for name, key, _gate in _stall_candidates(monitor) if name == step
    )


def reset_monitor():
    return {
        "state": "WAITING",
        "started_at": None,
        # These procedure steps are outside the reliable scope of the current
        # NanoOWL box detector and therefore require explicit human confirmation.
        "wet_confirmed": False,
        "rinse_confirmed": False,
        "dry_confirmed": False,
        "faucet_confirmed": False,
        # Two hands seen.
        "armed": False,
        # Initial 5-second confirmation timer.
        "confirmation_time": 0.0,
        # Washing has officially started.
        "rubbing_confirmed": False,
        # Total valid rubbing.
        "rubbing_time": 0.0,
        # Soap/foam seen at least once.
        "soap_seen": False,
        "soap_evidence_seconds": 0.0,
        # Continuous-water timer driving automatic wet-hands confirmation.
        "water_wet_seconds": 0.0,
        # Two-phase latch driving automatic rinse confirmation: water seen
        # again after rubbing, then water gone ("stopping the sink").
        "rinse_water_seen": False,
        "water_seen_seconds": 0.0,
        "water_gone_seconds": 0.0,
        # Continuous-seconds timer driving automatic dry confirmation.
        "towel_seconds": 0.0,
        # Separation timer.
        "separated_since": None,
        # Which fallback-eligible checkpoint (if any) is currently the
        # blocker, and since when - drives the "press W/N/D/F" nudge.
        "waiting_step": None,
        "waiting_since": None,
        # Coached WHO technique-step coaching (see advance_technique_variation):
        # which step we're on, its starting hand-grip signature, and whether
        # the grip has visibly changed since that step began.
        "technique_step_index": None,
        "technique_signature": None,
        "technique_variation_seen": False,
        # None / CORRECT / INCORRECT / PRACTICE.
        "result": None,
        "result_reason": "",
        "result_started": 0.0,
    }


# =========================================================
# ON-DEVICE CALIBRATION
# =========================================================

# Every number above that's plausibly worth tuning per-camera/per-sink,
# captured as {name: default} so load_config() below can validate a
# config.json against it. This is calibration only - it reads settings
# in, it never writes anything, and it has no bearing on the "nothing is
# persisted" rule for session/detection data.
TUNABLE_DEFAULTS = {
    "REQUIRED_RUB_TIME": REQUIRED_RUB_TIME,
    "MINIMUM_WASH_TIME": MINIMUM_WASH_TIME,
    "MAXIMUM_WASH_TIME": MAXIMUM_WASH_TIME,
    "INITIAL_CONFIRMATION_TIME": INITIAL_CONFIRMATION_TIME,
    "MAX_SEPARATION_TIME": MAX_SEPARATION_TIME,
    "RESULT_DISPLAY_TIME": RESULT_DISPLAY_TIME,
    "BOX_CONNECTION_PADDING": BOX_CONNECTION_PADDING,
    "MAX_FOREARM_GAP": MAX_FOREARM_GAP,
    "ROI_PADDING": ROI_PADDING,
    "TOWEL_FAUCET_PADDING": TOWEL_FAUCET_PADDING,
    "MOTION_PIXEL_THRESHOLD": MOTION_PIXEL_THRESHOLD,
    "MINIMUM_MOTION_RATIO": MINIMUM_MOTION_RATIO,
    "WATER_WET_SECONDS": WATER_WET_SECONDS,
    "WATER_CONFIRMATION_SECONDS": WATER_CONFIRMATION_SECONDS,
    "WATER_ABSENCE_SECONDS": WATER_ABSENCE_SECONDS,
    "TOWEL_CONFIRMATION_SECONDS": TOWEL_CONFIRMATION_SECONDS,
    "SOAP_CONFIRMATION_SECONDS": SOAP_CONFIRMATION_SECONDS,
    "HAND_PROXIMITY_PADDING": HAND_PROXIMITY_PADDING,
    "STALL_HINT_SECONDS": STALL_HINT_SECONDS,
    "GRIP_VARIATION_TOLERANCE": GRIP_VARIATION_TOLERANCE,
    "CAMERA_ID": CAMERA_ID,
    "CAMERA_WIDTH": CAMERA_WIDTH,
    "CAMERA_HEIGHT": CAMERA_HEIGHT,
    "CAMERA_RETRY_SECONDS": CAMERA_RETRY_SECONDS,
    "DETECTION_THRESHOLD": DETECTION_THRESHOLD,
    "SOAP_EVIDENCE_THRESHOLD": SOAP_EVIDENCE_THRESHOLD,
    "WATER_EVIDENCE_THRESHOLD": WATER_EVIDENCE_THRESHOLD,
    "TOWEL_EVIDENCE_THRESHOLD": TOWEL_EVIDENCE_THRESHOLD,
    "MAX_INFERENCE_FPS": MAX_INFERENCE_FPS,
}

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def load_config(path=None):
    """Apply on-device calibration overrides from a local JSON file.

    Reads {"WATER_EVIDENCE_THRESHOLD": 0.22, ...} and overwrites the
    matching module-level defaults above, so nanoowl_monitor.py (and this
    module's own functions) pick up the tuned values. Missing file, empty
    file, or unknown keys are all handled quietly rather than crashing the
    app over a calibration typo - this is the one file this module ever
    reads, and it never writes one.

    Must run before nanoowl_monitor.py does `from nanoowl_logic import
    THRESHOLD_NAME`, since that copies the value at that moment - see the
    import order at the top of nanoowl_monitor.py.
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
        globals()[name] = value
        applied[name] = value
    return applied
