import cv2
import math
import sys
import time
import numpy as np
import PIL.Image

from nanoowl.tree import Tree
from nanoowl.tree_predictor import TreePredictor
from nanoowl.owl_predictor import OwlPredictor

# Apply any on-device calibration from config.json BEFORE importing the
# specific threshold names below - `from nanoowl_logic import X` copies
# the value at that moment, so load_config()'s overrides must land first
# or they'd silently be ignored here. See nanoowl_logic.load_config.
import nanoowl_logic

nanoowl_logic.load_config()

from nanoowl_logic import (
    REQUIRED_RUB_TIME,
    MINIMUM_WASH_TIME,
    MAXIMUM_WASH_TIME,
    INITIAL_CONFIRMATION_TIME,
    MAX_SEPARATION_TIME,
    RESULT_DISPLAY_TIME,
    BOX_CONNECTION_PADDING,
    MAX_FOREARM_GAP,
    ROI_PADDING,
    MINIMUM_MOTION_RATIO,
    TOWEL_FAUCET_PADDING,
    HAND_PROXIMITY_PADDING,
    CAMERA_ID,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_RETRY_SECONDS,
    DETECTION_THRESHOLD,
    SOAP_EVIDENCE_THRESHOLD,
    WATER_EVIDENCE_THRESHOLD,
    TOWEL_EVIDENCE_THRESHOLD,
    WATER_WET_SECONDS,
    TOWEL_CONFIRMATION_SECONDS,
    boxes_connected,
    box_center,
    box_near_any,
    evidence_box_reasonable,
    hand_at_faucet,
    union_box,
    motion_ratio,
    detection_name,
    strongest_by_side,
    technique_prompt,
    procedure_elapsed,
    missing_checkpoints,
    reset_monitor,
    advance_wet_evidence,
    advance_rinse_evidence,
    advance_dry_evidence,
    advance_faucet_evidence,
    advance_soap_evidence,
    advance_technique_variation,
    technique_variation_due,
    advance_stall_tracking,
    stall_hint,
)

# =========================================================
# CONFIGURATION
# =========================================================
# Camera ID/resolution and detection thresholds now live in
# nanoowl_logic.py's TUNABLE_DEFAULTS, calibrated via config.json rather
# than edited here. What's left below is genuinely fixed per mode, not a
# calibration knob.

# Room mode exercises real hand/forearm tracking without requiring soap.
ROOM_MODE = "--room" in sys.argv

ENGINE_PATH = "/opt/nanoowl/data/" "owl_image_encoder_patch32.engine"

# Real sink test asks NanoOWL to also watch for running water and a towel,
# so wet/rinse/dry/tap-closed can be auto-detected instead of key presses.
# Room mode has no water/soap step at all, so it stays hands+forearms only.
PROMPT = (
    "[a left hand, a right hand, " "a left forearm, a right forearm]"
    if ROOM_MODE
    else "[a left hand, a right hand, "
    "a left forearm, a right forearm, soap foam, "
    "running water, a towel, a faucet]"
)


# =========================================================
# BRAND PALETTE (BGR, since that's what cv2 draws in)
# =========================================================
# Same four-color identity as bubbles/camera.py's Room hand test HUD and
# the argus-launcher desktop app - dark navy card, mint primary, cyan
# secondary, coral alert - so the real sink test doesn't look like a
# different, rougher product next to the rest of Argus.

PANEL_BG = (13, 39, 43)  # dark teal-navy card background
PANEL_BORDER = (70, 96, 94)  # subtle card border
MINT = (167, 208, 69)  # primary accent: hands, success, "go"
CYAN = (232, 199, 105)  # secondary accent: forearms, logo mark
AMBER = (90, 190, 245)  # evidence accent: soap, water, towel
CORAL = (95, 100, 235)  # alert accent: incomplete, missing, "stop"
FIXTURE = (150, 150, 150)  # muted: background fixtures like the faucet
TEXT_BRIGHT = (235, 248, 244)  # primary text, near-white mint tint
TEXT_MUTED = (202, 221, 217)  # secondary text
TEXT_FAINT = (140, 168, 165)  # tertiary / hint text
DIM = (110, 128, 126)  # "not yet" indicator


# =========================================================
# IMAGE HELPERS
# =========================================================


def cv2_to_pil(frame):

    return PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# =========================================================
# DRAW DETECTION BOX
# =========================================================


def draw_box(frame, box, label, color):

    x1, y1, x2, y2 = [int(value) for value in box]

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.putText(
        frame, label, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2
    )


# =========================================================
# RESULT SCREEN
# =========================================================


def result_screen(frame, correct, reason="", title=None):

    overlay = np.zeros_like(frame)
    overlay[:] = MINT if correct else CORAL

    output = cv2.addWeighted(frame, 0.25, overlay, 0.75, 0)

    if title is None:
        title = "OBSERVED STEPS COMPLETE" if correct else "HANDWASH INCOMPLETE"

    cv2.putText(
        output,
        title,
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        TEXT_BRIGHT,
        3,
        cv2.LINE_AA,
    )

    if reason:

        cv2.putText(
            output,
            reason,
            (30, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            TEXT_BRIGHT,
            2,
            cv2.LINE_AA,
        )

    return output


# =========================================================
# COMPACT CHECKLIST
# =========================================================


def _status_dot(frame, center, complete):
    """A filled mint dot with a checkmark cut-out when done, an empty dim
    ring otherwise - the same status-dot language bubbles/camera.py uses
    for HANDS READY / SOAP SEEN, instead of bracket text."""

    if complete:
        cv2.circle(frame, center, 7, MINT, -1, cv2.LINE_AA)
        cx, cy = center
        cv2.line(frame, (cx - 4, cy), (cx - 1, cy + 3), PANEL_BG, 2, cv2.LINE_AA)
        cv2.line(frame, (cx - 1, cy + 3), (cx + 4, cy - 3), PANEL_BG, 2, cv2.LINE_AA)
    else:
        cv2.circle(frame, center, 7, DIM, 2, cv2.LINE_AA)


def _progress_bar(frame, top_left, size, fraction, fill_color):
    x, y = top_left
    width, bar_height = size
    cv2.rectangle(frame, (x, y), (x + width, y + bar_height), PANEL_BORDER, -1)
    fill_width = int(width * max(0.0, min(1.0, fraction)))
    if fill_width:
        cv2.rectangle(frame, (x, y), (x + fill_width, y + bar_height), fill_color, -1)


def draw_checklist(frame, monitor, separation_elapsed, current_time):

    # Compact top-left card, styled to match the rest of the Argus product.

    x = 12
    y = 12

    width = min(500, frame.shape[1] - 24)
    height = 390

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), PANEL_BG, -1)
    frame[y : y + height, x : x + width] = cv2.addWeighted(
        overlay[y : y + height, x : x + width],
        0.88,
        frame[y : y + height, x : x + width],
        0.12,
        0,
    )
    cv2.rectangle(frame, (x, y), (x + width, y + height), PANEL_BORDER, 1, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (x + 4, y + height), MINT, -1)  # brand accent edge

    # -----------------------------------------------------
    # HEADER: logo mark + wordmark, mode badge
    # -----------------------------------------------------

    cv2.circle(frame, (x + 24, y + 30), 7, MINT, -1, cv2.LINE_AA)
    cv2.circle(frame, (x + 33, y + 24), 4, CYAN, -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        "ARGUS",
        (x + 45, y + 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        TEXT_BRIGHT,
        1,
        cv2.LINE_AA,
    )

    badge_text = "ROOM PRACTICE" if ROOM_MODE else "REAL SINK TEST"
    (badge_w, badge_h), _ = cv2.getTextSize(
        badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1
    )
    badge_right = x + width - 12
    badge_left = badge_right - badge_w - 16
    cv2.rectangle(
        frame, (badge_left, y + 16), (badge_right, y + 16 + badge_h + 10), CYAN, -1
    )
    cv2.putText(
        frame,
        badge_text,
        (badge_left + 8, y + 16 + badge_h + 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        PANEL_BG,
        1,
        cv2.LINE_AA,
    )

    subtitle = (
        "Practice only - soap and water bypassed, not WHO-verified"
        if ROOM_MODE
        else "WHO-guided handwash - 40 to 60 seconds"
    )
    cv2.putText(
        frame,
        subtitle,
        (x + 14, y + 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        TEXT_FAINT,
        1,
        cv2.LINE_AA,
    )
    cv2.line(
        frame, (x + 14, y + 68), (x + width - 14, y + 68), PANEL_BORDER, 1, cv2.LINE_AA
    )

    # -----------------------------------------------------
    # CHECKLIST ITEMS
    # -----------------------------------------------------

    checklist = [
        (monitor["armed"], "Two hands detected"),
        (
            monitor["wet_confirmed"],
            "Wet hands (press W)"
            if ROOM_MODE
            else "Wet hands (auto: water 3s - press W if missed)",
        ),
        (
            ROOM_MODE or monitor["soap_seen"],
            "Product bypassed - practice only" if ROOM_MODE else "Soap / foam observed",
        ),
        (monitor["rubbing_time"] >= REQUIRED_RUB_TIME, "30 sec active rubbing"),
        (
            monitor["rinse_confirmed"],
            "Rinsed (press N)"
            if ROOM_MODE
            else "Rinsed (auto: tap off - press N if missed)",
        ),
        (
            monitor["dry_confirmed"],
            "Single-use towel (press D)"
            if ROOM_MODE
            else "Single-use towel (auto - press D if missed)",
        ),
        (
            monitor["faucet_confirmed"],
            (
                "Tap closed with towel (press F)"
                if ROOM_MODE
                else "Tap closed with towel (auto - press F if missed)"
            ),
        ),
    ]

    start_y = y + 90

    for index, (complete, text) in enumerate(checklist):

        line_y = start_y + index * 23

        _status_dot(frame, (x + 22, line_y - 5), complete)

        cv2.putText(
            frame,
            text,
            (x + 40, line_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            TEXT_BRIGHT if complete else TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )

    # -----------------------------------------------------
    # RUBBING PROGRESS
    # -----------------------------------------------------

    progress_y = start_y + len(checklist) * 23 + 12

    rubbing_text = (
        f"Active rub {monitor['rubbing_time']:.1f}/{REQUIRED_RUB_TIME:.0f}s"
    )
    total_text = (
        f"Total {procedure_elapsed(monitor, current_time):.1f}/{MINIMUM_WASH_TIME:.0f}s"
    )
    cv2.putText(
        frame,
        rubbing_text,
        (x + 14, progress_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        TEXT_MUTED,
        1,
        cv2.LINE_AA,
    )
    (total_w, _), _ = cv2.getTextSize(total_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    cv2.putText(
        frame,
        total_text,
        (x + width - 14 - total_w, progress_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        TEXT_MUTED,
        1,
        cv2.LINE_AA,
    )

    _progress_bar(
        frame,
        (x + 14, progress_y + 8),
        (width - 28, 8),
        monitor["rubbing_time"] / REQUIRED_RUB_TIME,
        MINT,
    )

    # -----------------------------------------------------
    # RUB COUNTDOWN TIMER
    # -----------------------------------------------------
    # A large, bold countdown next to the small progress-bar readout above,
    # so remaining rub time is readable at a glance from sink distance
    # instead of only as fine print.

    timer_y = progress_y + 32
    remaining_rub_time = max(0.0, REQUIRED_RUB_TIME - monitor["rubbing_time"])

    if not monitor["rubbing_confirmed"]:
        timer_value, timer_color = f"{REQUIRED_RUB_TIME:.0f}s", TEXT_FAINT
    elif remaining_rub_time <= 0:
        timer_value, timer_color = "DONE", MINT
    elif monitor["state"] == "CONTACT_NO_MOTION":
        timer_value, timer_color = f"{remaining_rub_time:.0f}s", CORAL
    else:
        timer_value, timer_color = f"{remaining_rub_time:.0f}s", AMBER

    cv2.putText(
        frame,
        "RUB TIMER",
        (x + 14, timer_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        TEXT_MUTED,
        1,
        cv2.LINE_AA,
    )
    (timer_label_w, _), _ = cv2.getTextSize(
        "RUB TIMER", cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1
    )
    cv2.putText(
        frame,
        timer_value,
        (x + 14 + timer_label_w + 12, timer_y + 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        timer_color,
        2,
        cv2.LINE_AA,
    )

    # -----------------------------------------------------
    # COACHING / NEXT STEP / SEPARATION WARNING
    # -----------------------------------------------------

    guide_y = timer_y + 28

    if monitor["rubbing_confirmed"] and monitor["rubbing_time"] < REQUIRED_RUB_TIME:
        guide_text = "GUIDE: " + technique_prompt(monitor["rubbing_time"])
        if technique_variation_due(monitor, monitor["rubbing_time"]):
            guide_text += "  - try a different grip"
        cv2.putText(
            frame,
            guide_text,
            (x + 14, guide_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            CYAN,
            2,
            cv2.LINE_AA,
        )
    elif monitor["rubbing_time"] >= REQUIRED_RUB_TIME:
        remaining = missing_checkpoints(monitor, ROOM_MODE)
        next_step = remaining[0] if remaining else "finish the 40 second procedure"
        cv2.putText(
            frame,
            "NEXT: " + next_step,
            (x + 14, guide_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            CYAN,
            1,
            cv2.LINE_AA,
        )

    if (
        monitor["separated_since"] is not None
        and monitor["rubbing_time"] < REQUIRED_RUB_TIME
    ):
        separation_text = f"Apart {separation_elapsed:.1f}/{MAX_SEPARATION_TIME:.0f}s"
        (sep_w, _), _ = cv2.getTextSize(
            separation_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
        )
        cv2.putText(
            frame,
            separation_text,
            (x + width - 14 - sep_w, guide_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            CORAL,
            1,
            cv2.LINE_AA,
        )

    # -----------------------------------------------------
    # STALL / STUCK HINT
    # -----------------------------------------------------
    # Tells the operator when to reach for the manual fallback instead of
    # leaving them guessing why the checklist stopped advancing.

    hint_y = guide_y + 24

    if monitor["state"] == "CONTACT_NO_MOTION":
        cv2.putText(
            frame,
            "Keep moving - contact detected, not enough motion yet",
            (x + 14, hint_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            CORAL,
            1,
            cv2.LINE_AA,
        )
    else:
        hint = stall_hint(monitor, current_time)
        if hint is not None:
            step_name, key = hint
            cv2.putText(
                frame,
                f"Still waiting on {step_name} - press {key} if it was missed",
                (x + 14, hint_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                AMBER,
                1,
                cv2.LINE_AA,
            )

    # -----------------------------------------------------
    # CONTROLS
    # -----------------------------------------------------

    controls = (
        "W wet   N rinse   D dry   F faucet+towel   R reset   C calibrate   Q quit"
        if ROOM_MODE
        else "W/N/D/F = fallback only, auto normally   R reset   C calibrate   Q quit"
    )
    cv2.putText(
        frame,
        controls,
        (x + 14, y + height - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.33,
        TEXT_FAINT,
        1,
        cv2.LINE_AA,
    )


# =========================================================
# CALIBRATION OVERLAY
# =========================================================


def draw_calibration(
    frame,
    soap_score,
    water_score,
    towel_score,
    motion_ratio_value,
    contact,
    rubbing,
    wet_timer,
    rinse_latch,
    dry_seconds,
    towel_faucet,
    hand_faucet,
):
    """Live, on-screen-only readout of raw detection scores against their
    thresholds, toggled with C. Nothing here is written anywhere - it
    exists purely so config.json's thresholds can be tuned against what
    the camera is actually seeing, instead of guessed blind."""

    height, width = frame.shape[:2]
    right = min(width - 14, 900)
    top = height - 76

    overlay = frame.copy()
    cv2.rectangle(overlay, (14, top), (right, height - 14), PANEL_BG, -1)
    frame[top : height - 14, 14:right] = cv2.addWeighted(
        overlay[top : height - 14, 14:right],
        0.88,
        frame[top : height - 14, 14:right],
        0.12,
        0,
    )
    cv2.rectangle(frame, (14, top), (right, height - 14), PANEL_BORDER, 1, cv2.LINE_AA)

    cv2.putText(
        frame,
        "CALIBRATION - C to hide",
        (24, top + 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        MINT,
        1,
        cv2.LINE_AA,
    )

    line1 = (
        f"soap {soap_score:.2f}/{SOAP_EVIDENCE_THRESHOLD:.2f}   "
        f"water {water_score:.2f}/{WATER_EVIDENCE_THRESHOLD:.2f}   "
        f"towel {towel_score:.2f}/{TOWEL_EVIDENCE_THRESHOLD:.2f}   "
        f"motion {motion_ratio_value:.3f}/{MINIMUM_MOTION_RATIO:.3f}"
    )
    line2 = (
        f"contact {'Y' if contact else 'N'}   "
        f"rubbing {'Y' if rubbing else 'N'}   "
        f"wet-timer {wet_timer:.1f}/{WATER_WET_SECONDS:.1f}s   "
        f"rinse-latch {'Y' if rinse_latch else 'N'}   "
        f"dry-timer {dry_seconds:.1f}/{TOWEL_CONFIRMATION_SECONDS:.1f}s   "
        f"towel-faucet {'Y' if towel_faucet else 'N'}   "
        f"hand-faucet {'Y' if hand_faucet else 'N'}"
    )
    cv2.putText(
        frame, line1, (24, top + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_MUTED, 1, cv2.LINE_AA
    )
    cv2.putText(
        frame, line2, (24, top + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_MUTED, 1, cv2.LINE_AA
    )


def open_camera():
    """Open CAMERA_ID with the backend/resolution/buffer setup the sink
    test needs, shared between startup and the reconnect-on-failure path
    in main()'s loop so they can't drift out of sync. Returns a
    VideoCapture that may or may not be open - caller checks isOpened()."""

    # Explicit V4L2 backend, not OpenCV's default choice for this index.
    # On-device probing showed the default (GStreamer) backend silently
    # rejects CAP_PROP_BUFFERSIZE - set() returns False and get() reads
    # back 0 - so the "always the newest frame" behavior below never
    # actually applied. V4L2 accepts it (verified: set() returns True,
    # get() reads back 1) and opens this same camera successfully. Falls
    # back to the default backend if V4L2 can't open the device at all
    # (e.g. a CSI camera that only exposes itself through a GStreamer
    # pipeline), so this doesn't strand hardware V4L2 can't reach.
    camera = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)

    if not camera.isOpened():
        camera = cv2.VideoCapture(CAMERA_ID)

    if not camera.isOpened():
        return camera

    # A capped resolution keeps every per-frame cost (color conversion,
    # PIL conversion, NanoOWL preprocessing, display) proportional to a
    # sane frame size instead of whatever high-res default the driver
    # picks. A buffer size of 1 stops the driver from queuing up frames
    # while we're busy on inference, so camera.read() always returns the
    # newest frame instead of a growing backlog of stale ones.
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return camera


def main():

    # =========================================================
    # LOAD NANOOWL
    # =========================================================

    print("Loading NanoOWL...")

    predictor = TreePredictor(
        owl_predictor=OwlPredictor(image_encoder_engine=(ENGINE_PATH))
    )

    tree = Tree.from_prompt(PROMPT)

    clip_encodings = predictor.encode_clip_text(tree)

    owl_encodings = predictor.encode_owl_text(tree)

    print("NanoOWL loaded.")

    print("Prompt:", PROMPT)

    # =========================================================
    # OPEN CAMERA
    # =========================================================

    print("Opening camera...")

    camera = open_camera()

    if not camera.isOpened():

        print(f"Could not open " f"/dev/video{CAMERA_ID}")

        print("Try changing " "CAMERA_ID to 1.")

        raise SystemExit

    # =========================================================
    # MAIN VARIABLES
    # =========================================================

    monitor = reset_monitor()

    previous_gray = None

    last_frame_time = time.monotonic()

    # Set the moment camera.read() first starts failing in a row; cleared
    # the moment it succeeds again. Drives the reconnect-and-retry below
    # instead of a single dropped frame ending the whole session.
    camera_failure_since = None

    # Live calibration readout, toggled with C - shows raw detection scores
    # so the thresholds in config.json can be tuned against what the camera
    # is actually seeing, instead of guessed blind. Nothing here is saved.
    calibration = False

    # =========================================================
    # MAIN LOOP
    # =========================================================

    try:
        while True:

            # =====================================================
            # READ CAMERA
            # =====================================================

            success, frame = camera.read()

            if not success:

                if camera_failure_since is None:
                    camera_failure_since = time.monotonic()
                    print("Camera read failed - attempting to reconnect...")

                if time.monotonic() - camera_failure_since >= CAMERA_RETRY_SECONDS:
                    print(
                        f"Camera unreachable for {CAMERA_RETRY_SECONDS:.0f}s - giving up."
                    )
                    break

                camera.release()
                camera = open_camera()
                continue

            camera_failure_since = None

            # =====================================================
            # TIME
            # =====================================================

            current_time = time.monotonic()

            delta_time = min(current_time - last_frame_time, 0.5)

            last_frame_time = current_time

            # =====================================================
            # GRAYSCALE FOR MOTION
            # =====================================================

            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # =====================================================
            # NANOOWL PREDICTION
            # =====================================================

            output = predictor.predict(
                cv2_to_pil(frame),
                tree=tree,
                threshold=(DETECTION_THRESHOLD),
                clip_text_encodings=(clip_encodings),
                owl_text_encodings=(owl_encodings),
            )

            # =====================================================
            # COLLECT DETECTIONS
            # =====================================================

            hands = []

            forearms = []

            soap_detections = []

            water_detections = []

            towel_detections = []

            faucet_detections = []

            for detection in output.detections:

                if detection.parent_id != 0:
                    continue

                box = tuple(float(value) for value in detection.box)

                if box[2] <= box[0] or box[3] <= box[1]:
                    continue

                label = detection_name(detection, tree)

                if len(detection.scores) > 0:

                    score = float(detection.scores[-1])

                else:

                    score = 0.0

                item = {"box": box, "label": label, "score": score}

                label_lower = label.lower()

                # -------------------------------------------------
                # SOAP / FOAM
                # -------------------------------------------------
                # Soap/water/towel are amorphous/transparent enough that
                # NanoOWL will occasionally sweep a box across most of the
                # frame instead of the real thing (background, reflections,
                # a wet countertop) - evidence_box_reasonable drops those
                # before they can ever be drawn or scored, instead of
                # showing an oversized box and/or a false "detected" signal.

                if "soap" in label_lower or "foam" in label_lower:

                    if evidence_box_reasonable(box, frame.shape):
                        soap_detections.append(item)

                # -------------------------------------------------
                # RUNNING WATER
                # -------------------------------------------------

                elif "water" in label_lower:

                    if evidence_box_reasonable(box, frame.shape):
                        water_detections.append(item)

                # -------------------------------------------------
                # TOWEL
                # -------------------------------------------------

                elif "towel" in label_lower:

                    if evidence_box_reasonable(box, frame.shape):
                        towel_detections.append(item)

                # -------------------------------------------------
                # FAUCET
                # -------------------------------------------------

                elif "faucet" in label_lower:

                    faucet_detections.append(item)

                # -------------------------------------------------
                # FOREARMS
                # -------------------------------------------------

                elif "forearm" in label_lower:

                    forearms.append(item)

                # -------------------------------------------------
                # HANDS
                # -------------------------------------------------

                elif "hand" in label_lower:

                    hands.append(item)

            # =====================================================
            # SELECT HANDS / FOREARMS
            # =====================================================
            # Selected before soap/water evidence below so those can be
            # required to actually be near the hands/forearms in frame,
            # not just present anywhere in the picture.

            left_hand, right_hand = strongest_by_side(hands)

            left_forearm, right_forearm = strongest_by_side(forearms)

            selected_hands = [
                item for item in (left_hand, right_hand) if item is not None
            ]

            selected_forearms = [
                item for item in (left_forearm, right_forearm) if item is not None
            ]

            hand_boxes = [item["box"] for item in selected_hands]

            forearm_boxes = [item["box"] for item in selected_forearms]

            # Skipped (never required) when no hand/forearm box is visible
            # at all this frame, so momentary occlusion of the hands can't
            # itself break wet/rinse/soap detection.
            active_boxes = hand_boxes + forearm_boxes

            # =====================================================
            # DRAW DETECTIONS
            # =====================================================

            if left_hand:

                draw_box(frame, left_hand["box"], "LEFT HAND", MINT)

            if right_hand:

                draw_box(frame, right_hand["box"], "RIGHT HAND", MINT)

            if left_forearm:

                draw_box(frame, left_forearm["box"], "LEFT FOREARM", CYAN)

            if right_forearm:

                draw_box(frame, right_forearm["box"], "RIGHT FOREARM", CYAN)

            # =====================================================
            # SOAP / FOAM
            # =====================================================

            soap_detection = max(
                soap_detections, key=lambda item: item["score"], default=None
            )

            soap_evidence = (
                soap_detection is not None
                and soap_detection["score"] >= SOAP_EVIDENCE_THRESHOLD
                and monitor["armed"]
                and monitor["wet_confirmed"]
                and (
                    not active_boxes
                    or box_near_any(
                        soap_detection["box"], active_boxes, HAND_PROXIMITY_PADDING
                    )
                )
            )

            if advance_soap_evidence(monitor, soap_evidence, delta_time):
                print("Soap / foam observed near the hands.")

            if soap_detection is not None:

                draw_box(
                    frame,
                    soap_detection["box"],
                    (
                        "SOAP / FOAM?"
                        if not monitor["soap_seen"]
                        else "SOAP / FOAM OBSERVED"
                    ),
                    AMBER,
                )

            # =====================================================
            # WATER / TOWEL / FAUCET
            # =====================================================
            # Raw per-frame evidence for the automatic wet/rinse/dry/tap-off
            # checkpoints. The actual confirmation timing/debounce lives in
            # nanoowl_logic's advance_*_evidence functions, called after
            # hand/forearm geometry and rubbing time are settled below.

            # Faucet is picked first so hand-at-faucet contact can feed
            # into water evidence below: NanoOWL is much more reliable at
            # a solid object like a faucet than at classifying the
            # transparent running-water stream itself, so a hand actually
            # reaching the faucet is treated as water evidence in its own
            # right, not only the "running water" label.
            faucet_detection = max(
                faucet_detections, key=lambda item: item["score"], default=None
            )
            faucet_box = faucet_detection["box"] if faucet_detection is not None else None
            hand_faucet_contact = hand_at_faucet(faucet_box, active_boxes)

            water_detection = max(
                water_detections, key=lambda item: item["score"], default=None
            )
            water_label_evidence = (
                water_detection is not None
                and water_detection["score"] >= WATER_EVIDENCE_THRESHOLD
                and monitor["armed"]
                and (
                    not active_boxes
                    or box_near_any(
                        water_detection["box"], active_boxes, HAND_PROXIMITY_PADDING
                    )
                )
            )
            water_evidence = water_label_evidence or (
                monitor["armed"] and hand_faucet_contact
            )
            if water_detection is not None:
                draw_box(frame, water_detection["box"], "WATER", AMBER)

            towel_detection = max(
                towel_detections, key=lambda item: item["score"], default=None
            )
            towel_evidence = (
                towel_detection is not None
                and towel_detection["score"] >= TOWEL_EVIDENCE_THRESHOLD
            )
            if towel_detection is not None:
                draw_box(frame, towel_detection["box"], "TOWEL", AMBER)

            # Faucet box itself flips to the "evidence" amber and gets a
            # connecting line to the nearest hand/forearm while a hand is
            # at it, so it's visually obvious on screen *why* water is
            # being counted even when the running-water stream itself
            # isn't being picked up.
            if faucet_detection is not None:
                draw_box(
                    frame,
                    faucet_detection["box"],
                    "FAUCET - HAND AT TAP" if hand_faucet_contact else "FAUCET",
                    AMBER if hand_faucet_contact else FIXTURE,
                )
                if hand_faucet_contact:
                    faucet_center = box_center(faucet_detection["box"])
                    nearest_box = min(
                        active_boxes,
                        key=lambda box: math.hypot(
                            box_center(box)[0] - faucet_center[0],
                            box_center(box)[1] - faucet_center[1],
                        ),
                    )
                    cv2.line(
                        frame,
                        tuple(int(v) for v in box_center(nearest_box)),
                        tuple(int(v) for v in faucet_center),
                        AMBER,
                        2,
                        cv2.LINE_AA,
                    )

            towel_faucet_contact = (
                towel_evidence
                and faucet_detection is not None
                and boxes_connected(
                    towel_detection["box"],
                    faucet_detection["box"],
                    TOWEL_FAUCET_PADDING,
                )
            )

            # =====================================================
            # HAND GEOMETRY
            # =====================================================

            two_hands_visible = len(hand_boxes) == 2

            hands_connected = two_hands_visible and boxes_connected(
                hand_boxes[0], hand_boxes[1], BOX_CONNECTION_PADDING
            )

            # =====================================================
            # FOREARM GEOMETRY
            # =====================================================

            two_forearms_visible = len(forearm_boxes) == 2

            forearms_close = two_forearms_visible and boxes_connected(
                forearm_boxes[0], forearm_boxes[1], MAX_FOREARM_GAP
            )

            # =====================================================
            # ARM SESSION
            # =====================================================

            if two_hands_visible and not monitor["armed"]:

                monitor["armed"] = True

                # WHO's 40-60s window times the wash procedure itself,
                # starting at "wet hands" (step 1). Room mode has no wet
                # step and never claims WHO compliance, so its practice
                # timer starts as soon as hands are seen instead.
                if ROOM_MODE:
                    monitor["started_at"] = current_time

                monitor["state"] = "TWO_HANDS_READY"

                print("Two hands detected. " "Session armed.")

            # =====================================================
            # MOTION REGION
            # =====================================================

            visible_boxes = hand_boxes + forearm_boxes

            activity_roi = union_box(visible_boxes, frame.shape, ROI_PADDING)

            changed_ratio = motion_ratio(current_gray, previous_gray, activity_roi)

            movement_detected = changed_ratio >= MINIMUM_MOTION_RATIO

            # Small subtle motion ROI.
            if activity_roi is not None:

                x1, y1, x2, y2 = activity_roi

                cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), 1)

            # =====================================================
            # CONTACT DETECTION
            # =====================================================

            # Case 1:
            # Two hand boxes remain visible
            # and connect.
            #
            # Case 2:
            # Hands merge into one NanoOWL box
            # but two close forearms remain visible.

            merged_contact = (
                monitor["armed"] and len(hand_boxes) == 1 and forearms_close
            )

            contact_detected = monitor["armed"] and (hands_connected or merged_contact)

            # =====================================================
            # AUTOMATIC WET CONFIRMATION (WHO step 1)
            # =====================================================
            # Must run before VALID RUBBING below, since rubbing validity
            # depends on wet_confirmed in the real (non-room) sink test.

            if advance_wet_evidence(monitor, water_evidence, delta_time):
                if not ROOM_MODE:
                    monitor["started_at"] = current_time
                print("Auto-detected: hands wetted (water seen 3s).")

            # =====================================================
            # VALID RUBBING
            # =====================================================

            valid_rubbing = (
                contact_detected
                and movement_detected
                and (ROOM_MODE or (monitor["wet_confirmed"] and monitor["soap_seen"]))
            )

            # =====================================================
            # INITIAL 5 SECOND CONFIRMATION
            # =====================================================

            if monitor["result"] is None and not monitor["rubbing_confirmed"]:

                if valid_rubbing:

                    monitor["confirmation_time"] += delta_time

                    monitor["state"] = "CONFIRMING_RUBBING"

                    # ---------------------------------------------
                    # FIVE CONTINUOUS SECONDS REACHED
                    # ---------------------------------------------

                    if monitor["confirmation_time"] >= INITIAL_CONFIRMATION_TIME:

                        monitor["confirmation_time"] = INITIAL_CONFIRMATION_TIME

                        monitor["rubbing_confirmed"] = True

                        # Initial five seconds
                        # count toward total.
                        monitor["rubbing_time"] = INITIAL_CONFIRMATION_TIME

                        monitor["state"] = "CONFIRMED_RUBBING"

                        print("Rubbing confirmed " "after 5 continuous seconds.")

                else:

                    # Initial five seconds
                    # must be continuous.
                    monitor["confirmation_time"] = 0.0

                    if monitor["armed"]:

                        monitor["state"] = "TWO_HANDS_READY"

                    else:

                        monitor["state"] = "WAITING"

            # =====================================================
            # AFTER INITIAL 5 SECOND CONFIRMATION
            # =====================================================

            elif (
                monitor["result"] is None
                and monitor["rubbing_confirmed"]
                and monitor["rubbing_time"] < REQUIRED_RUB_TIME
            ):

                # =================================================
                # HANDS TOGETHER
                # =================================================

                if contact_detected:

                    # Hands returned before
                    # separation reached 5 seconds.
                    monitor["separated_since"] = None

                    if valid_rubbing:

                        monitor["rubbing_time"] += delta_time

                        monitor["state"] = "CONFIRMED_RUBBING"

                    else:

                        # Still touching,
                        # but not enough motion.
                        monitor["state"] = "CONTACT_NO_MOTION"

                # =================================================
                # HANDS SEPARATED
                # =================================================

                else:

                    if monitor["separated_since"] is None:

                        monitor["separated_since"] = current_time

                        print("Hands separated. " "Starting separation timer.")

                    separation_time = current_time - monitor["separated_since"]

                    monitor["state"] = "HANDS_SEPARATED"

                    # ---------------------------------------------
                    # PROLONGED SEPARATION -> INCOMPLETE
                    # ---------------------------------------------

                    if separation_time >= MAX_SEPARATION_TIME:

                        monitor["result"] = "INCORRECT"

                        monitor["result_reason"] = (
                            "Hands separated "
                            f"for more than {MAX_SEPARATION_TIME:.0f} seconds"
                        )

                        monitor["result_started"] = current_time

                        monitor["state"] = "INCOMPLETE"

                        print(
                            "INCORRECT: "
                            "hands remained separated "
                            f"for {MAX_SEPARATION_TIME:.0f} seconds."
                        )

            # =====================================================
            # AUTOMATIC RINSE / DRY / FAUCET CONFIRMATION
            # (WHO steps 9-11) - runs now that rubbing_time is settled
            # for this frame.
            # =====================================================

            rub_target_reached = monitor["rubbing_time"] >= REQUIRED_RUB_TIME

            if advance_rinse_evidence(
                monitor, water_evidence, rub_target_reached, delta_time
            ):
                print("Auto-detected: rinse complete (water stopped).")

            if advance_dry_evidence(monitor, towel_evidence, delta_time):
                print("Auto-detected: towel in use - hands dried.")

            if advance_faucet_evidence(monitor, towel_faucet_contact):
                print("Auto-detected: towel touched faucet - tap closed with towel.")

            # =====================================================
            # 40-60 SECOND PROCEDURE WINDOW
            # =====================================================

            if (
                monitor["result"] is None
                and monitor["started_at"] is not None
                and procedure_elapsed(monitor, current_time) > MAXIMUM_WASH_TIME
            ):
                monitor["state"] = "INCOMPLETE"
                monitor["result"] = "INCORRECT"
                missing = missing_checkpoints(monitor, ROOM_MODE)
                monitor["result_reason"] = (
                    "60s elapsed; missing: " + ", ".join(missing[:2])
                    if missing
                    else "Procedure exceeded the 40-60s window"
                )
                monitor["result_started"] = current_time
                print("INCOMPLETE:", monitor["result_reason"])

            # =====================================================
            # WHO-GUIDED FINAL VALIDATION
            # =====================================================

            if (
                monitor["result"] is None
                and monitor["rubbing_confirmed"]
                and monitor["rubbing_time"] >= REQUIRED_RUB_TIME
            ):

                # Stop the active-rubbing timer at the coached sequence duration.
                monitor["rubbing_time"] = REQUIRED_RUB_TIME

                elapsed = procedure_elapsed(monitor, current_time)
                ready = (
                    not missing_checkpoints(monitor, ROOM_MODE)
                    and elapsed >= MINIMUM_WASH_TIME
                )

                if ROOM_MODE and elapsed >= MINIMUM_WASH_TIME:
                    monitor["state"] = "PRACTICE_COMPLETE"
                    monitor["result"] = "PRACTICE"
                    monitor["result_started"] = current_time
                    print("PRACTICE COMPLETE: not recorded as WHO-verified hygiene.")
                elif ready:
                    monitor["state"] = "COMPLETE"
                    monitor["result"] = "CORRECT"
                    monitor["result_started"] = current_time
                    print("OBSERVED STEPS COMPLETE: WHO-guided 40-60 second workflow.")
                else:
                    monitor["state"] = "AWAITING_CHECKPOINTS"

            # =====================================================
            # SEPARATION TIMER
            # =====================================================

            if monitor["separated_since"] is not None:

                separation_elapsed = current_time - monitor["separated_since"]

            else:

                separation_elapsed = 0.0

            # =====================================================
            # TECHNIQUE-VARIATION COACHING (non-blocking nudge)
            # =====================================================

            if monitor["rubbing_confirmed"] and monitor["rubbing_time"] < REQUIRED_RUB_TIME:
                advance_technique_variation(monitor, monitor["rubbing_time"], hand_boxes)

            # =====================================================
            # STALL TRACKING (nudge toward the W/N/D/F fallback)
            # =====================================================

            advance_stall_tracking(monitor, current_time)

            # =====================================================
            # COMPACT CHECKLIST
            # =====================================================

            draw_checklist(frame, monitor, separation_elapsed, current_time)

            if calibration:

                draw_calibration(
                    frame,
                    soap_detection["score"] if soap_detection is not None else 0.0,
                    water_detection["score"] if water_detection is not None else 0.0,
                    towel_detection["score"] if towel_detection is not None else 0.0,
                    changed_ratio,
                    contact_detected,
                    valid_rubbing,
                    monitor["water_wet_seconds"],
                    monitor["rinse_water_seen"],
                    monitor["towel_seconds"],
                    towel_faucet_contact,
                    hand_faucet_contact,
                )

            # =====================================================
            # RESULT SCREEN
            # =====================================================

            if monitor["result"] == "INCORRECT":

                frame = result_screen(frame, False, monitor["result_reason"])

            elif monitor["result"] == "CORRECT":

                frame = result_screen(
                    frame, True, "Guided sequence + required checkpoints"
                )

            elif monitor["result"] == "PRACTICE":
                frame = result_screen(
                    frame,
                    True,
                    "No product or hygiene outcome verified",
                    title="ROOM PRACTICE COMPLETE",
                )

            # =====================================================
            # SHOW CAMERA
            # =====================================================

            cv2.imshow("Argus - NanoOWL Hand Hygiene", frame)

            key = cv2.waitKey(1) & 0xFF

            # =====================================================
            # QUIT
            # =====================================================

            if key == ord("q"):

                break

            # =====================================================
            # MANUAL RESET
            # =====================================================

            if key == ord("r"):

                monitor = reset_monitor()

                previous_gray = None

            # =====================================================
            # CALIBRATION TOGGLE
            # =====================================================

            if key == ord("c"):

                calibration = not calibration

                print("Monitor reset.")

            # W/N/D/F are a manual FALLBACK only. Wet/rinse/dry/faucet-closed
            # are normally auto-detected from water/towel/faucet evidence
            # above; these keys exist so a session isn't stuck if detection
            # misses on-device (lighting, camera angle, etc).
            if (
                key == ord("w")
                and monitor["result"] is None
                and monitor["armed"]
                and not monitor["rubbing_confirmed"]
                and not monitor["wet_confirmed"]
            ):
                monitor["wet_confirmed"] = True
                if not ROOM_MODE:
                    # WHO step 1: this is when the timed 40-60s procedure
                    # actually begins, not whenever hands first appeared.
                    monitor["started_at"] = current_time
                print("Manual fallback: hands wetted.")

            if (
                key == ord("n")
                and monitor["result"] is None
                and monitor["rubbing_time"] >= REQUIRED_RUB_TIME
                and (ROOM_MODE or monitor["soap_seen"])
                and not monitor["rinse_confirmed"]
            ):
                monitor["rinse_confirmed"] = True
                print("Manual fallback: hands rinsed.")

            if (
                key == ord("d")
                and monitor["result"] is None
                and monitor["rinse_confirmed"]
                and not monitor["dry_confirmed"]
            ):
                monitor["dry_confirmed"] = True
                print("Manual fallback: single-use towel drying.")

            if (
                key == ord("f")
                and monitor["result"] is None
                and monitor["dry_confirmed"]
                and not monitor["faucet_confirmed"]
            ):
                monitor["faucet_confirmed"] = True
                print("Manual fallback: faucet closed with towel.")

            # =====================================================
            # AUTOMATIC RESET AFTER RESULT
            # =====================================================

            if (
                monitor["result"] is not None
                and (current_time - monitor["result_started"]) >= RESULT_DISPLAY_TIME
            ):

                print("Result displayed for " "5 seconds.")

                print("Resetting entire " "handwash session.")

                monitor = reset_monitor()

                previous_gray = None

            previous_gray = current_gray

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
