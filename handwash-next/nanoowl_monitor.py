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
    box_near_any,
    union_box,
    motion_ratio,
    detection_name,
    strongest_by_side,
    procedure_elapsed,
    missing_checkpoints,
    reset_monitor,
    advance_wet_evidence,
    advance_rinse_evidence,
    advance_dry_evidence,
    advance_faucet_evidence,
    advance_soap_evidence,
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
    "a left forearm, a right forearm, foam, "
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
# ROUNDED-RECT PRIMITIVE
# =========================================================
# Every panel/chip in this file used to be a hard-cornered cv2.rectangle,
# which reads as a placeholder/wireframe rather than a finished product.
# One shared helper (filled or outline, via `thickness`) keeps every card
# in the HUD consistent instead of each caller approximating its own.


def rounded_rect(frame, top_left, bottom_right, radius, color, thickness=-1):

    x1, y1 = top_left
    x2, y2 = bottom_right
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))

    if thickness < 0:
        cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, -1, cv2.LINE_AA)
        for cx, cy in (
            (x1 + radius, y1 + radius),
            (x2 - radius, y1 + radius),
            (x1 + radius, y2 - radius),
            (x2 - radius, y2 - radius),
        ):
            cv2.circle(frame, (cx, cy), radius, color, -1, cv2.LINE_AA)
    else:
        for (cx, cy), start_angle in (
            ((x1 + radius, y1 + radius), 180),
            ((x2 - radius, y1 + radius), 270),
            ((x2 - radius, y2 - radius), 0),
            ((x1 + radius, y2 - radius), 90),
        ):
            cv2.ellipse(
                frame, (cx, cy), (radius, radius), 0, start_angle, start_angle + 90,
                color, thickness, cv2.LINE_AA,
            )
        cv2.line(frame, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)


def panel_with_shadow(frame, top_left, bottom_right, radius, fill_color, border_color, opacity=0.96):
    """Blends a soft dark shadow + a rounded, near-opaque card into `frame`
    in place - the shared look behind the checklist card and the
    calibration strip, instead of each caller hand-rolling its own
    addWeighted blend and flat-cornered rectangle."""

    x1, y1 = top_left
    x2, y2 = bottom_right
    pad = radius + 6

    shadow_region = (
        max(0, y1 - pad), min(frame.shape[0], y2 + pad + 6),
        max(0, x1 - pad), min(frame.shape[1], x2 + pad + 6),
    )
    sy1, sy2, sx1, sx2 = shadow_region
    shadow_layer = frame[sy1:sy2, sx1:sx2].copy()
    rounded_rect(
        shadow_layer, (x1 - sx1 + 3, y1 - sy1 + 5), (x2 - sx1 + 3, y2 - sy1 + 5),
        radius, (0, 0, 0), -1,
    )
    frame[sy1:sy2, sx1:sx2] = cv2.addWeighted(
        shadow_layer, 0.35, frame[sy1:sy2, sx1:sx2], 0.65, 0
    )

    card_layer = frame[y1:y2, x1:x2].copy()
    rounded_rect(card_layer, (0, 0), (x2 - x1, y2 - y1), radius, fill_color, -1)
    frame[y1:y2, x1:x2] = cv2.addWeighted(
        card_layer, opacity, frame[y1:y2, x1:x2], 1 - opacity, 0
    )
    rounded_rect(frame, (x1, y1), (x2, y2), radius, border_color, 1)


def pulsing_dot(frame, center, base_color, current_time, period=1.6, min_radius=3, max_radius=5):
    """A soft breathing dot - the little visual tell that this is a live
    model running right now, not a static screenshot. Cheap sine pulse on
    radius; alpha-blended so the glow can bleed past the dot itself."""

    phase = (math.sin(current_time * (2 * math.pi / period)) + 1) / 2
    radius = int(min_radius + (max_radius - min_radius) * phase)

    x, y = center
    reach = max_radius + 4
    gx1, gy1 = max(0, x - reach), max(0, y - reach)
    gx2, gy2 = x + reach, y + reach
    glow_region = frame[gy1:gy2, gx1:gx2]
    if glow_region.size:
        glow_layer = glow_region.copy()
        cv2.circle(glow_layer, (x - gx1, y - gy1), reach, base_color, -1, cv2.LINE_AA)
        frame[gy1:gy2, gx1:gx2] = cv2.addWeighted(
            glow_layer, 0.12 + 0.1 * phase, glow_region, 0.88 - 0.1 * phase, 0
        )

    cv2.circle(frame, center, radius, base_color, -1, cv2.LINE_AA)


# =========================================================
# DRAW DETECTION BOX
# =========================================================


def draw_box(frame, box, label, color):
    """An anti-aliased box plus a filled label chip (rather than bare
    stroked text) so the tag stays legible over any background - a busy
    sink, a bright window, skin tone - the way modern CV demo overlays
    (chips with a solid backing) read instead of a debug-print box."""

    x1, y1, x2, y2 = [int(value) for value in box]

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
    chip_y2 = max(text_h + 14, y1 - 4)
    chip_y1 = chip_y2 - text_h - 12
    # Clamped so a box near the right edge doesn't push its label chip
    # off-frame - the box itself may still run off-screen, but the tag
    # naming it should always stay readable.
    chip_x1 = max(0, min(x1, frame.shape[1] - text_w - 16))
    chip_x2 = chip_x1 + text_w + 16

    rounded_rect(frame, (chip_x1, chip_y1), (chip_x2, chip_y2), 5, color, -1)
    cv2.putText(
        frame, label, (chip_x1 + 8, chip_y2 - 7), cv2.FONT_HERSHEY_SIMPLEX,
        0.46, PANEL_BG, 1, cv2.LINE_AA,
    )


# =========================================================
# RESULT SCREEN
# =========================================================


_RESULT_ICON_RADIUS = 46


def _draw_check_icon(frame, center, color, scale=1.0):
    cx, cy = center
    radius = int(_RESULT_ICON_RADIUS * scale)
    cv2.circle(frame, center, radius, color, 4, cv2.LINE_AA)
    p1 = (cx - int(radius * 0.45), cy + int(radius * 0.05))
    p2 = (cx - int(radius * 0.12), cy + int(radius * 0.38))
    p3 = (cx + int(radius * 0.48), cy - int(radius * 0.35))
    cv2.line(frame, p1, p2, color, 6, cv2.LINE_AA)
    cv2.line(frame, p2, p3, color, 6, cv2.LINE_AA)


def _draw_cross_icon(frame, center, color, scale=1.0):
    cx, cy = center
    radius = int(_RESULT_ICON_RADIUS * scale)
    arm = int(32 * scale)
    cv2.circle(frame, center, radius, color, 4, cv2.LINE_AA)
    cv2.line(frame, (cx - arm, cy - arm), (cx + arm, cy + arm), color, 6, cv2.LINE_AA)
    cv2.line(frame, (cx - arm, cy + arm), (cx + arm, cy - arm), color, 6, cv2.LINE_AA)


def result_screen(frame, correct, reason="", title=None, elapsed=None):
    """Full-frame result takeover. `elapsed` (seconds since the result
    fired) drives a quick fade/scale-in on the icon and a wash-in on the
    color overlay so the result lands as a distinct beat instead of
    popping in fully-formed on the very first frame."""

    fade = 1.0 if elapsed is None else min(1.0, elapsed / 0.35)
    accent = MINT if correct else CORAL

    overlay = np.zeros_like(frame)
    overlay[:] = accent

    output = cv2.addWeighted(frame, 1 - 0.75 * fade, overlay, 0.75 * fade, 0)

    if title is None:
        title = "OBSERVED STEPS COMPLETE" if correct else "HANDWASH INCOMPLETE"

    height, width = output.shape[:2]
    icon_center = (width // 2, max(110, height // 2 - 120))
    icon_scale = 0.5 + 0.5 * min(1.0, fade * 1.4)

    if correct:
        _draw_check_icon(output, icon_center, TEXT_BRIGHT, icon_scale)
    else:
        _draw_cross_icon(output, icon_center, TEXT_BRIGHT, icon_scale)

    (title_w, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.05, 2)
    cv2.putText(
        output,
        title,
        (max(30, (width - title_w) // 2), icon_center[1] + 90),
        cv2.FONT_HERSHEY_DUPLEX,
        1.05,
        TEXT_BRIGHT,
        2,
        cv2.LINE_AA,
    )

    if reason:

        (reason_w, _), _ = cv2.getTextSize(reason, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
        cv2.putText(
            output,
            reason,
            (max(30, (width - reason_w) // 2), icon_center[1] + 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            TEXT_BRIGHT,
            1,
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
        cv2.circle(frame, center, 4, MINT, -1, cv2.LINE_AA)
        cx, cy = center
        cv2.line(frame, (cx - 2, cy), (cx - 1, cy + 1), PANEL_BG, 1, cv2.LINE_AA)
        cv2.line(frame, (cx - 1, cy + 1), (cx + 2, cy - 2), PANEL_BG, 1, cv2.LINE_AA)
    else:
        cv2.circle(frame, center, 4, DIM, 1, cv2.LINE_AA)


def _progress_bar(frame, top_left, size, fraction, fill_color):
    x, y = top_left
    width, bar_height = size
    radius = bar_height // 2
    rounded_rect(frame, (x, y), (x + width, y + bar_height), radius, PANEL_BORDER, -1)
    fill_width = int(width * max(0.0, min(1.0, fraction)))
    if fill_width >= bar_height:
        rounded_rect(frame, (x, y), (x + fill_width, y + bar_height), radius, fill_color, -1)
    elif fill_width:
        cv2.circle(frame, (x + radius, y + radius), radius, fill_color, -1, cv2.LINE_AA)


def draw_checklist(frame, monitor, separation_elapsed, current_time):
    """Compact 224 x 184 card: about half the previous 310 x 260 area."""
    x, y = 12, 12
    width, height = min(224, frame.shape[1] - 24), 184
    font = cv2.FONT_HERSHEY_SIMPLEX

    def label(text, left, baseline, color=TEXT_MUTED, scale=0.31):
        # Fit long prompts and calibrated timing values inside the card.
        available = x + width - 9 - left
        text_width = cv2.getTextSize(text, font, scale, 1)[0][0]
        if text_width > available:
            scale *= available / text_width
        cv2.putText(frame, text, (left, baseline), font, scale, color, 1, cv2.LINE_AA)

    panel_with_shadow(frame, (x, y), (x + width, y + height), 10, PANEL_BG, PANEL_BORDER)
    rounded_rect(frame, (x, y + 8), (x + 2, y + height - 8), 1, MINT, -1)

    pulsing_dot(frame, (x + 12, y + 14), MINT, current_time, min_radius=2, max_radius=3)
    label("ARGUS", x + 22, y + 18, TEXT_BRIGHT, 0.36)
    badge = "ROOM PRACTICE" if ROOM_MODE else "REAL SINK"
    badge_width = cv2.getTextSize(badge, font, 0.27, 1)[0][0]
    badge_left = x + width - badge_width - 17
    rounded_rect(frame, (badge_left, y + 7), (x + width - 8, y + 23), 3, CYAN, -1)
    label(badge, badge_left + 4, y + 18, PANEL_BG, 0.27)
    subtitle = (
        "Practice only - soap/water bypassed"
        if ROOM_MODE
        else f"WHO-guided | {MINIMUM_WASH_TIME:.0f}-{MAXIMUM_WASH_TIME:.0f}s"
    )
    label(subtitle, x + 9, y + 33, TEXT_FAINT, 0.27)

    # Keep fallback keys beside their step without repeating instructions.
    checklist = [
        (monitor["armed"], "Two hands detected", ""),
        (monitor["wet_confirmed"], "Wet hands", "W"),
        (ROOM_MODE or monitor["soap_seen"],
         "Soap bypassed (practice)" if ROOM_MODE else "Foam observed", ""),
        (monitor["rubbing_time"] >= REQUIRED_RUB_TIME,
         f"{REQUIRED_RUB_TIME:.0f}s active rubbing", ""),
        (monitor["rinse_confirmed"], "Rinsed", "N"),
        (monitor["dry_confirmed"], "Single-use towel", "D"),
        (monitor["faucet_confirmed"], "Tap closed with towel", "F"),
    ]
    for index, (complete, text, key) in enumerate(checklist):
        baseline = y + 47 + index * 13
        _status_dot(frame, (x + 13, baseline - 3), complete)
        label(text, x + 24, baseline, TEXT_BRIGHT if complete else TEXT_MUTED)
        if key:
            label(key, x + width - 17, baseline, TEXT_FAINT, 0.27)

    label(f"Rub {monitor['rubbing_time']:.1f}/{REQUIRED_RUB_TIME:.0f}s",
          x + 9, y + 140, scale=0.29)
    total = f"Total {procedure_elapsed(monitor, current_time):.1f}/{MINIMUM_WASH_TIME:.0f}s"
    total_width = cv2.getTextSize(total, font, 0.29, 1)[0][0]
    label(total, x + width - 9 - total_width, y + 140, scale=0.29)
    _progress_bar(frame, (x + 9, y + 146), (width - 18, 4),
                  monitor["rubbing_time"] / REQUIRED_RUB_TIME, MINT)

    # One contextual line avoids overlapping coaching and separation text.
    hint = stall_hint(monitor, current_time)
    if monitor["separated_since"] is not None and monitor["rubbing_time"] < REQUIRED_RUB_TIME:
        guidance = f"Hands apart {separation_elapsed:.1f}/{MAX_SEPARATION_TIME:.0f}s"
        color = CORAL
    elif monitor["state"] == "CONTACT_NO_MOTION":
        guidance, color = "Keep moving your hands", CORAL
    elif hint is not None:
        step, key = hint
        guidance, color = f"{step}: press {key} if missed", AMBER
    elif monitor["rubbing_confirmed"] and monitor["rubbing_time"] < REQUIRED_RUB_TIME:
        guidance = f"Keep rubbing - {REQUIRED_RUB_TIME - monitor['rubbing_time']:.0f}s left"
        color = CYAN
    elif monitor["rubbing_time"] >= REQUIRED_RUB_TIME:
        remaining = missing_checkpoints(monitor, ROOM_MODE)
        guidance = "Next: " + (remaining[0] if remaining else f"finish {MINIMUM_WASH_TIME:.0f}s procedure")
        color = CYAN
    else:
        guidance = "W/N/D/F: manual steps" if ROOM_MODE else "Auto steps | W/N/D/F if missed"
        color = TEXT_FAINT
    label(guidance, x + 9, y + 163, color, 0.28)
    label("R reset   C calibrate   Q quit", x + 9, y + 177, TEXT_FAINT, 0.27)


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
):
    """Live, on-screen-only readout of raw detection scores against their
    thresholds, toggled with C. Nothing here is written anywhere - it
    exists purely so config.json's thresholds can be tuned against what
    the camera is actually seeing, instead of guessed blind."""

    height, width = frame.shape[:2]
    right = min(width - 14, 900)
    top = height - 76

    panel_with_shadow(frame, (14, top), (right, height - 14), 10, PANEL_BG, PANEL_BORDER, opacity=0.88)

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
        f"foam {soap_score:.2f}/{SOAP_EVIDENCE_THRESHOLD:.2f}   "
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
        f"towel-faucet {'Y' if towel_faucet else 'N'}"
    )
    cv2.putText(
        frame, line1, (24, top + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_MUTED, 1, cv2.LINE_AA
    )
    cv2.putText(
        frame, line2, (24, top + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_MUTED, 1, cv2.LINE_AA
    )


def fit_frame_to_window(frame, window_name):
    """cv2's own backend stretches the displayed frame to whatever size the
    window has been resized to, but does it with a cheap nearest-neighbor
    scale - fine at 1:1, visibly blocky once the window is bigger than the
    frame (dragged larger, maximized, a kiosk display). Doing that resize
    ourselves with real interpolation keeps the picture smooth at any
    window size instead of relying on that fallback."""

    try:
        _, _, window_width, window_height = cv2.getWindowImageRect(window_name)
    except cv2.error:
        return frame

    if window_width <= 0 or window_height <= 0:
        return frame

    frame_height, frame_width = frame.shape[:2]

    if (window_width, window_height) == (frame_width, frame_height):
        return frame

    interpolation = (
        cv2.INTER_AREA if window_width < frame_width else cv2.INTER_CUBIC
    )

    return cv2.resize(
        frame, (window_width, window_height), interpolation=interpolation
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

    # WINDOW_NORMAL (rather than the default WINDOW_AUTOSIZE) gives a
    # regular, resizable window with normal title-bar minimize/maximize
    # controls, sized to the window manager's default instead of being
    # forced fullscreen or to a fixed resolution.
    WINDOW_NAME = "Argus - NanoOWL Hand Hygiene"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

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

                if "soap" in label_lower or "foam" in label_lower:

                    soap_detections.append(item)

                # -------------------------------------------------
                # RUNNING WATER
                # -------------------------------------------------

                elif "water" in label_lower:

                    water_detections.append(item)

                # -------------------------------------------------
                # TOWEL
                # -------------------------------------------------

                elif "towel" in label_lower:

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
                print("Foam observed near the hands.")

            if soap_detection is not None:

                draw_box(
                    frame,
                    soap_detection["box"],
                    "FOAM?" if not monitor["soap_seen"] else "FOAM OBSERVED",
                    AMBER,
                )

            # =====================================================
            # WATER / TOWEL / FAUCET
            # =====================================================
            # Raw per-frame evidence for the automatic wet/rinse/dry/tap-off
            # checkpoints. The actual confirmation timing/debounce lives in
            # nanoowl_logic's advance_*_evidence functions, called after
            # hand/forearm geometry and rubbing time are settled below.

            water_detection = max(
                water_detections, key=lambda item: item["score"], default=None
            )
            water_evidence = (
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

            faucet_detection = max(
                faucet_detections, key=lambda item: item["score"], default=None
            )
            if faucet_detection is not None:
                draw_box(frame, faucet_detection["box"], "FAUCET", FIXTURE)

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

                cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), 1, cv2.LINE_AA)

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
                    f"{MAXIMUM_WASH_TIME:.0f}s elapsed; missing: " + ", ".join(missing[:2])
                    if missing
                    else f"Procedure exceeded the {MINIMUM_WASH_TIME:.0f}-{MAXIMUM_WASH_TIME:.0f}s window"
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
                    print(
                        "OBSERVED STEPS COMPLETE: WHO-guided "
                        f"{MINIMUM_WASH_TIME:.0f}-{MAXIMUM_WASH_TIME:.0f} second workflow."
                    )
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
                )

            # =====================================================
            # RESULT SCREEN
            # =====================================================

            if monitor["result"] is not None:
                result_elapsed = current_time - monitor["result_started"]

            if monitor["result"] == "INCORRECT":

                frame = result_screen(
                    frame, False, monitor["result_reason"], elapsed=result_elapsed
                )

            elif monitor["result"] == "CORRECT":

                frame = result_screen(
                    frame,
                    True,
                    "Guided sequence + required checkpoints",
                    elapsed=result_elapsed,
                )

            elif monitor["result"] == "PRACTICE":
                frame = result_screen(
                    frame,
                    True,
                    "No product or hygiene outcome verified",
                    title="ROOM PRACTICE COMPLETE",
                    elapsed=result_elapsed,
                )

            # =====================================================
            # SHOW CAMERA
            # =====================================================

            cv2.imshow(WINDOW_NAME, fit_frame_to_window(frame, WINDOW_NAME))

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
