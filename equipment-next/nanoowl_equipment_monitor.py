"""Argus equipment scan - "general" tier: NanoOWL camera capture, drawing,
and the live checklist loop.

Checks two things only, on purpose (see equipment-next/README.md for the
full scope caveat):
  1. Is the configured bare-minimum tray present (equipment_logic.
     REQUIRED_EQUIPMENT)?
  2. Does anything in frame look bloodstained or dirty, by a classical-CV
     color heuristic over each detected item's own box?

This is a visual screening aid, not a sterility verification and not a
certified clinical device. Like handwash-next/nanoowl_monitor.py, this
module needs the real NanoOWL/TensorRT container to import or run and is
therefore not covered by the unit test suite - equipment_logic.py (the
detection-agnostic checklist/contamination logic this file calls into) is
what's actually tested.

Nothing here is persisted: results are shown live on screen and discarded
- no CSV, no database, no network call. The only file this module reads
is an optional local config.json for calibration (see equipment_logic.
load_config); it never writes one.
"""

import time

import cv2
import PIL.Image

from nanoowl.tree import Tree
from nanoowl.tree_predictor import TreePredictor
from nanoowl.owl_predictor import OwlPredictor

# Apply any on-device calibration from config.json BEFORE importing the
# specific names below - `from equipment_logic import X` copies the value
# at that moment, so load_config()'s overrides must land first or they'd
# silently be ignored here. See equipment_logic.load_config.
import equipment_logic

equipment_logic.load_config()

from equipment_logic import (
    REQUIRED_EQUIPMENT,
    DETECTION_THRESHOLD,
    match_checklist,
    advance_scan,
    reset_scan,
    missing_items,
    contaminated_items,
    scan_passed,
)

# =========================================================
# CONFIGURATION
# =========================================================

CAMERA_ID = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_RETRY_SECONDS = 5.0

ENGINE_PATH = "/opt/nanoowl/data/owl_image_encoder_patch32.engine"

# NanoOWL prompt built from the configured checklist, e.g.
# "[a scalpel, a forceps, a scissors, a hemostat, a needle holder, a retractor]"
PROMPT = "[" + ", ".join(f"a {name}" for name in REQUIRED_EQUIPMENT) + "]"


# =========================================================
# BRAND PALETTE (BGR) - same identity as handwash-next/nanoowl_monitor.py
# =========================================================

PANEL_BG = (13, 39, 43)
PANEL_BORDER = (70, 96, 94)
MINT = (167, 208, 69)
CYAN = (232, 199, 105)
AMBER = (90, 190, 245)
CORAL = (95, 100, 235)
TEXT_BRIGHT = (235, 248, 244)
TEXT_MUTED = (202, 221, 217)
TEXT_FAINT = (140, 168, 165)
DIM = (110, 128, 126)


# =========================================================
# IMAGE HELPERS
# =========================================================


def cv2_to_pil(frame):
    return PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def draw_box(frame, box, label, color):
    x1, y1, x2, y2 = [int(value) for value in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame, label, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2
    )


def detection_name(detection, tree):
    if len(detection.labels) == 0:
        return "unknown"
    label_index = int(detection.labels[-1])
    if 0 <= label_index < len(tree.labels):
        return str(tree.labels[label_index])
    return "unknown"


# =========================================================
# CHECKLIST PANEL
# =========================================================


def _status_dot(frame, center, color, filled=True):
    if filled:
        cv2.circle(frame, center, 7, color, -1, cv2.LINE_AA)
    else:
        cv2.circle(frame, center, 7, color, 2, cv2.LINE_AA)


def _item_color(item_state):
    if item_state["blood_flagged"] or item_state["dirt_flagged"]:
        return CORAL
    if item_state["confirmed"]:
        return MINT
    return DIM


def _item_status_text(name, item_state):
    if item_state["blood_flagged"]:
        return f"{name} - possible bloodstain"
    if item_state["dirt_flagged"]:
        return f"{name} - looks dirty"
    if item_state["confirmed"]:
        return name
    return f"{name} - not seen"


def draw_checklist(frame, scan):
    x, y = 12, 12
    width = min(500, frame.shape[1] - 24)
    height = 60 + len(scan["items"]) * 23 + 60

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), PANEL_BG, -1)
    frame[y : y + height, x : x + width] = cv2.addWeighted(
        overlay[y : y + height, x : x + width], 0.88,
        frame[y : y + height, x : x + width], 0.12, 0,
    )
    cv2.rectangle(frame, (x, y), (x + width, y + height), PANEL_BORDER, 1, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (x + 4, y + height), MINT, -1)

    cv2.circle(frame, (x + 24, y + 30), 7, MINT, -1, cv2.LINE_AA)
    cv2.circle(frame, (x + 33, y + 24), 4, CYAN, -1, cv2.LINE_AA)
    cv2.putText(frame, "ARGUS", (x + 45, y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_BRIGHT, 1, cv2.LINE_AA)

    badge_text = "EQUIPMENT - GENERAL SCAN"
    (badge_w, badge_h), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
    badge_right = x + width - 12
    badge_left = badge_right - badge_w - 16
    cv2.rectangle(frame, (badge_left, y + 16), (badge_right, y + 16 + badge_h + 10), CYAN, -1)
    cv2.putText(
        frame, badge_text, (badge_left + 8, y + 16 + badge_h + 3),
        cv2.FONT_HERSHEY_SIMPLEX, 0.36, PANEL_BG, 1, cv2.LINE_AA,
    )

    cv2.putText(
        frame, "Baseline tray check + visual stain screen - not a sterility check",
        (x + 14, y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_FAINT, 1, cv2.LINE_AA,
    )
    cv2.line(frame, (x + 14, y + 68), (x + width - 14, y + 68), PANEL_BORDER, 1, cv2.LINE_AA)

    start_y = y + 90
    for index, (name, item_state) in enumerate(scan["items"].items()):
        line_y = start_y + index * 23
        color = _item_color(item_state)
        _status_dot(frame, (x + 22, line_y - 5), color, filled=(color != DIM))
        cv2.putText(
            frame, _item_status_text(name, item_state), (x + 40, line_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            TEXT_BRIGHT if color == MINT else TEXT_MUTED, 1, cv2.LINE_AA,
        )

    footer_y = start_y + len(scan["items"]) * 23 + 20
    passed = scan_passed(scan)
    result_text = "READY" if passed else "NOT READY"
    result_color = MINT if passed else CORAL
    cv2.putText(
        frame, result_text, (x + 14, footer_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, result_color, 2, cv2.LINE_AA,
    )

    controls = "R reset   Q quit"
    cv2.putText(
        frame, controls, (x + 14, y + height - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.33, TEXT_FAINT, 1, cv2.LINE_AA,
    )


def open_camera():
    camera = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
    if not camera.isOpened():
        camera = cv2.VideoCapture(CAMERA_ID)
    if not camera.isOpened():
        return camera

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera


def main():
    print("Loading NanoOWL...")
    predictor = TreePredictor(owl_predictor=OwlPredictor(image_encoder_engine=ENGINE_PATH))
    tree = Tree.from_prompt(PROMPT)
    clip_encodings = predictor.encode_clip_text(tree)
    owl_encodings = predictor.encode_owl_text(tree)
    print("NanoOWL loaded.")
    print("Prompt:", PROMPT)

    print("Opening camera...")
    camera = open_camera()
    if not camera.isOpened():
        print(f"Could not open /dev/video{CAMERA_ID}")
        print("Try changing CAMERA_ID to 1.")
        raise SystemExit

    scan = reset_scan()
    last_frame_time = time.monotonic()
    camera_failure_since = None

    try:
        while True:
            success, frame = camera.read()

            if not success:
                if camera_failure_since is None:
                    camera_failure_since = time.monotonic()
                    print("Camera read failed - attempting to reconnect...")
                if time.monotonic() - camera_failure_since >= CAMERA_RETRY_SECONDS:
                    print(f"Camera unreachable for {CAMERA_RETRY_SECONDS:.0f}s - giving up.")
                    break
                camera.release()
                camera = open_camera()
                continue

            camera_failure_since = None

            current_time = time.monotonic()
            delta_time = min(current_time - last_frame_time, 0.5)
            last_frame_time = current_time

            output = predictor.predict(
                cv2_to_pil(frame),
                tree=tree,
                threshold=DETECTION_THRESHOLD,
                clip_text_encodings=clip_encodings,
                owl_text_encodings=owl_encodings,
            )

            detections = []
            for detection in output.detections:
                if detection.parent_id != 0:
                    continue
                box = tuple(float(value) for value in detection.box)
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                label = detection_name(detection, tree)
                score = float(detection.scores[-1]) if len(detection.scores) > 0 else 0.0
                detections.append({"box": box, "label": label, "score": score})

            matches = match_checklist(detections)
            advance_scan(scan, matches, frame, delta_time)

            for name, item_state in scan["items"].items():
                if item_state["last_box"] is None:
                    continue
                color = _item_color(item_state)
                draw_box(frame, item_state["last_box"], name.upper(), color)

            draw_checklist(frame, scan)

            cv2.imshow("Argus - Equipment General Scan", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                scan = reset_scan()
                print("Scan reset.")

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
