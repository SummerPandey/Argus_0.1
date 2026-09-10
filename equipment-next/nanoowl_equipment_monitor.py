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
    scan_passed,
)
from equipment_hud import draw_checklist, draw_detection_box, item_kind, MINT, CORAL, DIM

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
# IMAGE HELPERS
# =========================================================


def cv2_to_pil(frame):
    return PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def detection_name(detection, tree):
    if len(detection.labels) == 0:
        return "unknown"
    label_index = int(detection.labels[-1])
    if 0 <= label_index < len(tree.labels):
        return str(tree.labels[label_index])
    return "unknown"


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
                kind = item_kind(item_state)
                color = CORAL if kind == "flagged" else (MINT if kind == "confirmed" else DIM)
                draw_detection_box(frame, item_state["last_box"], name.title().upper(), color)

            draw_checklist(frame, scan, ready=scan_passed(scan))

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
