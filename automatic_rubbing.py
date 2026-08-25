# NOTE: this is the original prototype, superseded by
# handwash-next/nanoowl_monitor.py (same NanoOWL approach, plus crash-safe
# cleanup, automatic wet/rinse/dry/faucet detection, WHO-timing fixes, and
# test coverage via handwash-next/nanoowl_logic.py). Kept for reference.
# See the top-level README.md for the current entry point.

import cv2
import time
import numpy as np
import PIL.Image

from nanoowl.tree import Tree
from nanoowl.tree_predictor import TreePredictor
from nanoowl.owl_predictor import OwlPredictor


# =========================================================
# CONFIGURATION
# =========================================================

CAMERA_ID = 0

ENGINE_PATH = (
    "/opt/nanoowl/data/"
    "owl_image_encoder_patch32.engine"
)

PROMPT = (
    "[a left hand, a right hand, "
    "a left forearm, a right forearm, "
    "soap foam]"
)

DETECTION_THRESHOLD = 0.10


# =========================================================
# HANDWASH RULES
# =========================================================

# Total valid rubbing required.
REQUIRED_RUB_TIME = 20.0

# Hands must initially remain together/rubbing
# for five continuous seconds.
INITIAL_CONFIRMATION_TIME = 5.0

# Once washing has been confirmed:
# hands may be separated for less than 5 seconds.
# Five full seconds apart -> RED.
MAX_SEPARATION_TIME = 5.0

# Red/green screen remains for 5 seconds.
# Then the entire session resets.
RESULT_DISPLAY_TIME = 5.0


# =========================================================
# DETECTION GEOMETRY
# =========================================================

# Small allowed gap between two hand boxes.
BOX_CONNECTION_PADDING = 35

# Allowed gap between forearms.
MAX_FOREARM_GAP = 120

# Padding around hand/forearm motion region.
ROI_PADDING = 25


# =========================================================
# MOTION SETTINGS
# =========================================================

MOTION_PIXEL_THRESHOLD = 20

MINIMUM_MOTION_RATIO = 0.025


# =========================================================
# IMAGE HELPERS
# =========================================================

def cv2_to_pil(frame):

    return PIL.Image.fromarray(
        cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )
    )


def box_center(box):

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0
    )


def boxes_connected(
    box1,
    box2,
    padding
):

    a_x1, a_y1, a_x2, a_y2 = box1
    b_x1, b_y1, b_x2, b_y2 = box2

    return (
        a_x1 <= b_x2 + padding
        and b_x1 <= a_x2 + padding
        and a_y1 <= b_y2 + padding
        and b_y1 <= a_y2 + padding
    )


def union_box(
    boxes,
    frame_shape,
    padding=0
):

    if not boxes:
        return None

    height, width = frame_shape[:2]

    x1 = max(
        0,
        int(
            min(
                box[0]
                for box in boxes
            ) - padding
        )
    )

    y1 = max(
        0,
        int(
            min(
                box[1]
                for box in boxes
            ) - padding
        )
    )

    x2 = min(
        width,
        int(
            max(
                box[2]
                for box in boxes
            ) + padding
        )
    )

    y2 = min(
        height,
        int(
            max(
                box[3]
                for box in boxes
            ) + padding
        )
    )

    if x2 <= x1 or y2 <= y1:
        return None

    return (
        x1,
        y1,
        x2,
        y2
    )


# =========================================================
# DRAW DETECTION BOX
# =========================================================

def draw_box(
    frame,
    box,
    label,
    color
):

    x1, y1, x2, y2 = [
        int(value)
        for value in box
    ]

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    cv2.putText(
        frame,
        label,
        (
            x1,
            max(
                25,
                y1 - 8
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        2
    )


# =========================================================
# NANOOWL LABEL HELPER
# =========================================================

def detection_name(
    detection,
    tree
):

    if len(detection.labels) == 0:

        return "unknown"

    label_index = int(
        detection.labels[-1]
    )

    if (
        0
        <= label_index
        < len(tree.labels)
    ):

        return str(
            tree.labels[
                label_index
            ]
        )

    return "unknown"


# =========================================================
# LEFT / RIGHT DETECTION HELPER
# =========================================================

def strongest_by_side(items):

    left_items = [
        item
        for item in items
        if "left"
        in item["label"].lower()
    ]

    right_items = [
        item
        for item in items
        if "right"
        in item["label"].lower()
    ]

    left = max(
        left_items,
        key=lambda item:
        item["score"],
        default=None
    )

    right = max(
        right_items,
        key=lambda item:
        item["score"],
        default=None
    )


    # =====================================================
    # FALLBACK
    # =====================================================

    # If NanoOWL does not consistently label
    # left/right, use the two strongest boxes
    # and determine sides from screen position.

    if (
        (
            left is None
            or right is None
        )
        and len(items) >= 2
    ):

        selected = sorted(
            items,
            key=lambda item:
            item["score"],
            reverse=True
        )[:2]

        selected.sort(
            key=lambda item:
            box_center(
                item["box"]
            )[0]
        )

        left, right = selected

    return left, right


# =========================================================
# MOTION DETECTION
# =========================================================

def motion_ratio(
    current_gray,
    previous_gray,
    roi
):

    if (
        previous_gray is None
        or roi is None
    ):
        return 0.0


    x1, y1, x2, y2 = roi


    current_roi = (
        current_gray[
            y1:y2,
            x1:x2
        ]
    )


    previous_roi = (
        previous_gray[
            y1:y2,
            x1:x2
        ]
    )


    if current_roi.size == 0:
        return 0.0


    if (
        previous_roi.shape
        != current_roi.shape
    ):
        return 0.0


    current_roi = cv2.GaussianBlur(
        current_roi,
        (7, 7),
        0
    )


    previous_roi = cv2.GaussianBlur(
        previous_roi,
        (7, 7),
        0
    )


    difference = cv2.absdiff(
        current_roi,
        previous_roi
    )


    changed = (
        difference
        >= MOTION_PIXEL_THRESHOLD
    )


    return (
        float(
            np.count_nonzero(
                changed
            )
        )
        / changed.size
    )


# =========================================================
# RESULT SCREEN
# =========================================================

def result_screen(
    frame,
    correct,
    reason=""
):

    overlay = np.zeros_like(
        frame
    )


    if correct:

        overlay[:] = (
            0,
            180,
            0
        )

    else:

        overlay[:] = (
            0,
            0,
            255
        )


    output = cv2.addWeighted(
        frame,
        0.20,
        overlay,
        0.80,
        0
    )


    if correct:

        title = (
            "HANDWASH COMPLETE"
        )

    else:

        title = (
            "HANDWASH INCOMPLETE"
        )


    cv2.putText(
        output,
        title,
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        3
    )


    if reason:

        cv2.putText(
            output,
            reason,
            (30, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


    return output


# =========================================================
# COMPACT CHECKLIST
# =========================================================

def draw_checklist(
    frame,
    monitor,
    separation_elapsed
):

    # Compact top-left panel.

    x = 12
    y = 12

    width = 355
    height = 190


    overlay = frame.copy()


    # -----------------------------------------------------
    # TRANSPARENT BLACK BACKGROUND
    # -----------------------------------------------------

    cv2.rectangle(
        overlay,
        (x, y),
        (
            x + width,
            y + height
        ),
        (10, 10, 10),
        -1
    )


    frame[
        y:y + height,
        x:x + width
    ] = cv2.addWeighted(

        overlay[
            y:y + height,
            x:x + width
        ],

        0.70,

        frame[
            y:y + height,
            x:x + width
        ],

        0.30,

        0
    )


    # -----------------------------------------------------
    # BORDER
    # -----------------------------------------------------

    cv2.rectangle(
        frame,
        (x, y),
        (
            x + width,
            y + height
        ),
        (100, 100, 100),
        1
    )


    # -----------------------------------------------------
    # TITLE
    # -----------------------------------------------------

    cv2.putText(
        frame,
        "HANDWASH",
        (
            x + 14,
            y + 25
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        2
    )


    # -----------------------------------------------------
    # CHECKLIST ITEMS
    # -----------------------------------------------------

    checklist = [

        (
            monitor["armed"],
            "Two hands detected"
        ),

        (
            monitor[
                "rubbing_confirmed"
            ],
            "5 sec contact"
        ),

        (
            monitor[
                "soap_seen"
            ],
            "Soap / foam"
        ),

        (
            monitor[
                "rubbing_time"
            ]
            >= REQUIRED_RUB_TIME,
            "20 sec rubbing"
        )
    ]


    start_y = y + 53


    for index, (
        complete,
        text
    ) in enumerate(checklist):


        line_y = (
            start_y
            + index * 25
        )


        if complete:

            status = "[X]"

            color = (
                80,
                230,
                110
            )

        else:

            status = "[ ]"

            color = (
                200,
                200,
                200
            )


        cv2.putText(
            frame,
            status,
            (
                x + 14,
                line_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            color,
            2
        )


        cv2.putText(
            frame,
            text,
            (
                x + 53,
                line_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (240, 240, 240),
            1
        )


    # -----------------------------------------------------
    # RUBBING TIMER
    # -----------------------------------------------------

    rubbing_text = (
        f"Rubbing "
        f"{monitor['rubbing_time']:.1f}/"
        f"{REQUIRED_RUB_TIME:.0f}s"
    )


    cv2.putText(
        frame,
        rubbing_text,
        (
            x + 14,
            y + 166
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (220, 220, 220),
        1
    )


    # -----------------------------------------------------
    # SEPARATION TIMER
    # -----------------------------------------------------

    if (
        monitor[
            "separated_since"
        ]
        is not None
    ):

        separation_text = (
            f"Apart "
            f"{separation_elapsed:.1f}/"
            f"{MAX_SEPARATION_TIME:.0f}s"
        )


        cv2.putText(
            frame,
            separation_text,
            (
                x + 180,
                y + 166
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (220, 220, 220),
            1
        )


# =========================================================
# RESET MONITOR
# =========================================================

def reset_monitor():

    return {

        "state":
            "WAITING",

        # Two hands seen.
        "armed":
            False,

        # Initial 5-second confirmation timer.
        "confirmation_time":
            0.0,

        # Washing has officially started.
        "rubbing_confirmed":
            False,

        # Total valid rubbing.
        "rubbing_time":
            0.0,

        # Soap/foam seen at least once.
        "soap_seen":
            False,

        # Separation timer.
        "separated_since":
            None,

        # None / CORRECT / INCORRECT.
        "result":
            None,

        "result_reason":
            "",

        "result_started":
            0.0
    }


# =========================================================
# LOAD NANOOWL
# =========================================================

print(
    "Loading NanoOWL..."
)


predictor = TreePredictor(

    owl_predictor=OwlPredictor(

        image_encoder_engine=(
            ENGINE_PATH
        )
    )
)


tree = Tree.from_prompt(
    PROMPT
)


clip_encodings = (
    predictor.encode_clip_text(
        tree
    )
)


owl_encodings = (
    predictor.encode_owl_text(
        tree
    )
)


print(
    "NanoOWL loaded."
)

print(
    "Prompt:",
    PROMPT
)


# =========================================================
# OPEN CAMERA
# =========================================================

print(
    "Opening camera..."
)


camera = cv2.VideoCapture(
    CAMERA_ID
)


if not camera.isOpened():

    print(
        f"Could not open "
        f"/dev/video{CAMERA_ID}"
    )

    print(
        "Try changing "
        "CAMERA_ID to 1."
    )

    raise SystemExit


# =========================================================
# MAIN VARIABLES
# =========================================================

monitor = reset_monitor()

previous_gray = None

last_frame_time = (
    time.monotonic()
)


# =========================================================
# MAIN LOOP
# =========================================================

while True:


    # =====================================================
    # READ CAMERA
    # =====================================================

    success, frame = (
        camera.read()
    )


    if not success:

        print(
            "Could not read "
            "camera frame."
        )

        break


    # =====================================================
    # TIME
    # =====================================================

    current_time = (
        time.monotonic()
    )


    delta_time = min(

        current_time
        - last_frame_time,

        0.5
    )


    last_frame_time = (
        current_time
    )


    # =====================================================
    # GRAYSCALE FOR MOTION
    # =====================================================

    current_gray = (
        cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )
    )


    # =====================================================
    # NANOOWL PREDICTION
    # =====================================================

    output = predictor.predict(

        cv2_to_pil(
            frame
        ),

        tree=tree,

        threshold=(
            DETECTION_THRESHOLD
        ),

        clip_text_encodings=(
            clip_encodings
        ),

        owl_text_encodings=(
            owl_encodings
        )
    )


    # =====================================================
    # COLLECT DETECTIONS
    # =====================================================

    hands = []

    forearms = []

    soap_detections = []


    for detection in (
        output.detections
    ):


        if (
            detection.parent_id
            != 0
        ):
            continue


        box = tuple(

            float(value)

            for value
            in detection.box
        )


        if (
            box[2] <= box[0]
            or box[3] <= box[1]
        ):
            continue


        label = detection_name(
            detection,
            tree
        )


        if (
            len(
                detection.scores
            )
            > 0
        ):

            score = float(
                detection.scores[-1]
            )

        else:

            score = 0.0


        item = {

            "box":
                box,

            "label":
                label,

            "score":
                score
        }


        label_lower = (
            label.lower()
        )


        # -------------------------------------------------
        # SOAP / FOAM
        # -------------------------------------------------

        if (
            "soap"
            in label_lower
            or
            "foam"
            in label_lower
        ):

            soap_detections.append(
                item
            )


        # -------------------------------------------------
        # FOREARMS
        # -------------------------------------------------

        elif (
            "forearm"
            in label_lower
        ):

            forearms.append(
                item
            )


        # -------------------------------------------------
        # HANDS
        # -------------------------------------------------

        elif (
            "hand"
            in label_lower
        ):

            hands.append(
                item
            )


    # =====================================================
    # SOAP / FOAM
    # =====================================================

    soap_detection = max(

        soap_detections,

        key=lambda item:
        item["score"],

        default=None
    )


    if (
        soap_detection
        is not None
    ):


        if not monitor[
            "soap_seen"
        ]:

            print(
                "Soap / foam detected."
            )


        # Remember soap for the
        # remainder of this session.
        monitor[
            "soap_seen"
        ] = True


        draw_box(

            frame,

            soap_detection[
                "box"
            ],

            "SOAP / FOAM",

            (255, 180, 50)
        )


    # =====================================================
    # SELECT HANDS / FOREARMS
    # =====================================================

    (
        left_hand,
        right_hand
    ) = strongest_by_side(
        hands
    )


    (
        left_forearm,
        right_forearm
    ) = strongest_by_side(
        forearms
    )


    selected_hands = [

        item

        for item in (
            left_hand,
            right_hand
        )

        if item is not None
    ]


    selected_forearms = [

        item

        for item in (
            left_forearm,
            right_forearm
        )

        if item is not None
    ]


    hand_boxes = [

        item["box"]

        for item
        in selected_hands
    ]


    forearm_boxes = [

        item["box"]

        for item
        in selected_forearms
    ]


    # =====================================================
    # DRAW DETECTIONS
    # =====================================================

    if left_hand:

        draw_box(
            frame,
            left_hand["box"],
            "LEFT HAND",
            (255, 100, 0)
        )


    if right_hand:

        draw_box(
            frame,
            right_hand["box"],
            "RIGHT HAND",
            (0, 255, 255)
        )


    if left_forearm:

        draw_box(
            frame,
            left_forearm["box"],
            "LEFT FOREARM",
            (255, 0, 255)
        )


    if right_forearm:

        draw_box(
            frame,
            right_forearm["box"],
            "RIGHT FOREARM",
            (0, 200, 0)
        )


    # =====================================================
    # HAND GEOMETRY
    # =====================================================

    two_hands_visible = (
        len(hand_boxes)
        == 2
    )


    hands_connected = (

        two_hands_visible

        and boxes_connected(

            hand_boxes[0],

            hand_boxes[1],

            BOX_CONNECTION_PADDING
        )
    )


    # =====================================================
    # FOREARM GEOMETRY
    # =====================================================

    two_forearms_visible = (
        len(forearm_boxes)
        == 2
    )


    forearms_close = (

        two_forearms_visible

        and boxes_connected(

            forearm_boxes[0],

            forearm_boxes[1],

            MAX_FOREARM_GAP
        )
    )


    # =====================================================
    # ARM SESSION
    # =====================================================

    if (
        two_hands_visible
        and not monitor["armed"]
    ):

        monitor["armed"] = True

        monitor["state"] = (
            "TWO_HANDS_READY"
        )


        print(
            "Two hands detected. "
            "Session armed."
        )


    # =====================================================
    # MOTION REGION
    # =====================================================

    visible_boxes = (
        hand_boxes
        + forearm_boxes
    )


    activity_roi = union_box(

        visible_boxes,

        frame.shape,

        ROI_PADDING
    )


    changed_ratio = (
        motion_ratio(

            current_gray,

            previous_gray,

            activity_roi
        )
    )


    movement_detected = (

        changed_ratio
        >= MINIMUM_MOTION_RATIO
    )


    # Small subtle motion ROI.
    if (
        activity_roi
        is not None
    ):

        x1, y1, x2, y2 = (
            activity_roi
        )


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (160, 160, 160),
            1
        )


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

        monitor["armed"]

        and len(hand_boxes) == 1

        and forearms_close
    )


    contact_detected = (

        monitor["armed"]

        and (

            hands_connected

            or merged_contact
        )
    )


    # =====================================================
    # VALID RUBBING
    # =====================================================

    valid_rubbing = (

        contact_detected

        and movement_detected
    )


    # =====================================================
    # INITIAL 5 SECOND CONFIRMATION
    # =====================================================

    if (

        monitor["result"]
        is None

        and not monitor[
            "rubbing_confirmed"
        ]
    ):


        if valid_rubbing:


            monitor[
                "confirmation_time"
            ] += delta_time


            monitor["state"] = (
                "CONFIRMING_RUBBING"
            )


            # ---------------------------------------------
            # FIVE CONTINUOUS SECONDS REACHED
            # ---------------------------------------------

            if (

                monitor[
                    "confirmation_time"
                ]
                >= INITIAL_CONFIRMATION_TIME
            ):


                monitor[
                    "confirmation_time"
                ] = (
                    INITIAL_CONFIRMATION_TIME
                )


                monitor[
                    "rubbing_confirmed"
                ] = True


                # Initial five seconds
                # count toward total.
                monitor[
                    "rubbing_time"
                ] = (
                    INITIAL_CONFIRMATION_TIME
                )


                monitor["state"] = (
                    "CONFIRMED_RUBBING"
                )


                print(
                    "Rubbing confirmed "
                    "after 5 continuous seconds."
                )


        else:


            # Initial five seconds
            # must be continuous.
            monitor[
                "confirmation_time"
            ] = 0.0


            if monitor["armed"]:

                monitor["state"] = (
                    "TWO_HANDS_READY"
                )

            else:

                monitor["state"] = (
                    "WAITING"
                )


    # =====================================================
    # AFTER INITIAL 5 SECOND CONFIRMATION
    # =====================================================

    elif (

        monitor["result"]
        is None

        and monitor[
            "rubbing_confirmed"
        ]
    ):


        # =================================================
        # HANDS TOGETHER
        # =================================================

        if contact_detected:


            # Hands returned before
            # separation reached 5 seconds.
            monitor[
                "separated_since"
            ] = None


            if valid_rubbing:


                monitor[
                    "rubbing_time"
                ] += delta_time


                monitor["state"] = (
                    "CONFIRMED_RUBBING"
                )


            else:


                # Still touching,
                # but not enough motion.
                monitor["state"] = (
                    "CONTACT_NO_MOTION"
                )


        # =================================================
        # HANDS SEPARATED
        # =================================================

        else:


            if (
                monitor[
                    "separated_since"
                ]
                is None
            ):


                monitor[
                    "separated_since"
                ] = current_time


                print(
                    "Hands separated. "
                    "Starting 5-second timer."
                )


            separation_time = (

                current_time

                - monitor[
                    "separated_since"
                ]
            )


            monitor["state"] = (
                "HANDS_SEPARATED"
            )


            # ---------------------------------------------
            # FIVE SECONDS APART -> RED
            # ---------------------------------------------

            if (
                separation_time
                >= MAX_SEPARATION_TIME
            ):


                monitor[
                    "result"
                ] = "INCORRECT"


                monitor[
                    "result_reason"
                ] = (
                    "Hands separated "
                    "for more than 5 seconds"
                )


                monitor[
                    "result_started"
                ] = current_time


                monitor["state"] = (
                    "INCOMPLETE"
                )


                print(
                    "INCORRECT: "
                    "hands remained separated "
                    "for 5 seconds."
                )


    # =====================================================
    # 20 SECOND FINAL VALIDATION
    # =====================================================

    if (

        monitor["result"]
        is None

        and monitor[
            "rubbing_confirmed"
        ]

        and monitor[
            "rubbing_time"
        ] >= REQUIRED_RUB_TIME
    ):


        # Stop timer exactly at 20.
        monitor[
            "rubbing_time"
        ] = REQUIRED_RUB_TIME


        # =================================================
        # SOAP / FOAM DETECTED
        # =================================================

        if monitor[
            "soap_seen"
        ]:


            monitor["state"] = (
                "COMPLETE"
            )


            monitor[
                "result"
            ] = "CORRECT"


            monitor[
                "result_started"
            ] = current_time


            print(
                "SUCCESS: "
                "20 seconds completed "
                "and soap / foam was detected."
            )


        # =================================================
        # SOAP / FOAM NOT DETECTED
        # =================================================

        else:


            monitor["state"] = (
                "INCOMPLETE"
            )


            monitor[
                "result"
            ] = "INCORRECT"


            monitor[
                "result_reason"
            ] = (
                "20 seconds completed - "
                "NO SOAP / FOAM DETECTED"
            )


            monitor[
                "result_started"
            ] = current_time


            print(
                "INCORRECT: "
                "20 seconds completed "
                "without soap / foam."
            )


    # =====================================================
    # SEPARATION TIMER
    # =====================================================

    if (
        monitor[
            "separated_since"
        ]
        is not None
    ):


        separation_elapsed = (

            current_time

            - monitor[
                "separated_since"
            ]
        )


    else:

        separation_elapsed = 0.0


    # =====================================================
    # COMPACT CHECKLIST
    # =====================================================

    draw_checklist(

        frame,

        monitor,

        separation_elapsed
    )


    # =====================================================
    # RESULT SCREEN
    # =====================================================

    if (
        monitor["result"]
        == "INCORRECT"
    ):


        frame = result_screen(

            frame,

            False,

            monitor[
                "result_reason"
            ]
        )


    elif (
        monitor["result"]
        == "CORRECT"
    ):


        frame = result_screen(

            frame,

            True,

            "Soap detected + 20 seconds completed"
        )


    # =====================================================
    # SHOW CAMERA
    # =====================================================

    cv2.imshow(

        "Automatic Hand Rubbing Monitor",

        frame
    )


    key = (
        cv2.waitKey(1)
        & 0xFF
    )


    # =====================================================
    # QUIT
    # =====================================================

    if key == ord("q"):

        break


    # =====================================================
    # MANUAL RESET
    # =====================================================

    if key == ord("r"):


        monitor = (
            reset_monitor()
        )


        previous_gray = None


        print(
            "Monitor reset."
        )


    # =====================================================
    # AUTOMATIC RESET AFTER RESULT
    # =====================================================

    if (

        monitor["result"]
        is not None

        and (

            current_time

            - monitor[
                "result_started"
            ]

        ) >= RESULT_DISPLAY_TIME
    ):


        print(
            "Result displayed for "
            "5 seconds."
        )

        print(
            "Resetting entire "
            "handwash session."
        )


        monitor = (
            reset_monitor()
        )


        previous_gray = None


    previous_gray = (
        current_gray
    )


# =========================================================
# CLEANUP
# =========================================================

camera.release()

cv2.destroyAllWindows()