from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path

from .session import Observation


def clamp(value):
    return max(0.0, min(1.0, value))


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def palm_center(points):
    ids = (0, 5, 9, 13, 17)
    return (sum(points[i][0] for i in ids) / 5, sum(points[i][1] for i in ids) / 5)


def palm_scale(points):
    return max(.01, (distance(points[0], points[9]) + distance(points[5], points[17])) / 2)


@dataclass(frozen=True)
class VisionResult:
    observation: Observation
    landmarks: tuple
    contact: float
    motion: float
    reversals: int
    soap_signal: float
    forearms: tuple = tuple()
    forearm_contact: float = 0.0


class RubbingSignal:
    """Scale-invariant, temporally filtered hand-rubbing evidence."""

    def __init__(self):
        self.previous = None
        self.samples = deque(maxlen=18)

    def reset(self):
        self.previous = None
        self.samples.clear()

    def update(self, first, second):
        c1, c2 = palm_center(first), palm_center(second)
        scale = (palm_scale(first) + palm_scale(second)) / 2
        contact = clamp((2.8 - distance(c1, c2) / scale) / 1.35)
        relative = (c1[0] - c2[0], c1[1] - c2[1])
        speed = signed_motion = 0.0
        if self.previous:
            old_relative, old_scale = self.previous
            denominator = max(scale, old_scale, .01)
            dx = (relative[0] - old_relative[0]) / denominator
            dy = (relative[1] - old_relative[1]) / denominator
            speed = math.hypot(dx, dy)
            signed_motion = dx if abs(dx) >= abs(dy) else dy
        self.previous = (relative, scale)
        self.samples.append(signed_motion)
        active = [v for v in self.samples if abs(v) > .035]
        reversals = sum(a * b < 0 for a, b in zip(active, active[1:]))
        periodicity = clamp(reversals / 3)
        motion = clamp((speed - .018) / .14)
        return contact * motion * (.35 + .65 * periodicity), contact, motion, reversals


class FoamSignal:
    """Conservative bright, low-saturation foam evidence within the hand ROI."""

    def __init__(self):
        self.baseline = None
        self.history = deque(maxlen=12)

    def update(self, frame, points, cv2, np):
        height, width = frame.shape[:2]
        xs = [p[0] for hand in points for p in hand]
        ys = [p[1] for hand in points for p in hand]
        x1, x2 = int(max(0, min(xs)-.025)*width), int(min(1, max(xs)+.025)*width)
        y1, y2 = int(max(0, min(ys)-.025)*height), int(min(1, max(ys)+.025)*height)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = (hsv[:, :, 1] < 58) & (hsv[:, :, 2] > 175)
        ratio = float(np.count_nonzero(mask)) / mask.size
        self.baseline = ratio if self.baseline is None else min(self.baseline*.995 + ratio*.005, ratio)
        self.history.append(clamp((max(0.0, ratio-self.baseline)-.035)/.13))
        sustained = sum(v > .45 for v in self.history) >= 7
        return sum(self.history)/len(self.history) if sustained else 0.0


class LocalMotionSignal:
    """Measures internal motion in a hand-local, scale-normalized crop."""

    def __init__(self):
        self.previous = None

    def reset(self):
        self.previous = None

    def update(self, frame, points, cv2, np):
        height, width = frame.shape[:2]
        xs = [p[0] for hand in points for p in hand]
        ys = [p[1] for hand in points for p in hand]
        pad = .06
        x1, x2 = int(max(0, min(xs)-pad)*width), int(min(1, max(xs)+pad)*width)
        y1, y2 = int(max(0, min(ys)-pad)*height), int(min(1, max(ys)+pad)*height)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            self.reset()
            return 0.0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        score = 0.0
        if self.previous is not None:
            difference = cv2.absdiff(gray, self.previous)
            changed = float(np.count_nonzero(difference > 11)) / difference.size
            score = clamp((changed-.025)/.16)
        self.previous = gray
        return score


class HandVision:
    def __init__(self, mp):
        model_path = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"
        if not model_path.exists():
            raise RuntimeError("Missing hand model. Run: ./install-camera-deps.sh")
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=.62,
            min_hand_presence_confidence=.62,
            min_tracking_confidence=.62,
        )
        self.mp = mp
        self.model = mp.tasks.vision.HandLandmarker.create_from_options(options)
        pose_path = Path(__file__).resolve().parents[1] / "models" / "pose_landmarker_lite.task"
        self.pose_model = None
        if pose_path.exists():
            pose_options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(pose_path)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=.45,
                min_pose_presence_confidence=.45,
                min_tracking_confidence=.45,
                output_segmentation_masks=False,
            )
            self.pose_model = mp.tasks.vision.PoseLandmarker.create_from_options(pose_options)
        self.rubbing = RubbingSignal()
        self.foam = FoamSignal()
        self.local_motion = LocalMotionSignal()
        self.last_two_hands_at = -100.0
        self.pose_frame = 0
        self.forearms = tuple()
        self.forearm_contact = 0.0

    def close(self):
        self.model.close()
        if self.pose_model is not None:
            self.pose_model.close()

    def _update_forearms(self, image, timestamp_ms):
        if self.pose_model is None:
            return
        self.pose_frame += 1
        if self.pose_frame % 5:
            return
        poses = self.pose_model.detect_for_video(image, timestamp_ms).pose_landmarks or []
        if not poses:
            self.forearms, self.forearm_contact = tuple(), 0.0
            return
        pose = poses[0]
        # MediaPipe Pose: left elbow/wrist 13/15, right elbow/wrist 14/16.
        required = (13, 15, 14, 16)
        if any((pose[i].visibility or 0) < .35 for i in required):
            self.forearms, self.forearm_contact = tuple(), 0.0
            return
        left = ((pose[13].x, pose[13].y), (pose[15].x, pose[15].y))
        right = ((pose[14].x, pose[14].y), (pose[16].x, pose[16].y))
        arm_scale = max(.02, (distance(*left) + distance(*right)) / 2)
        wrist_gap = distance(left[1], right[1]) / arm_scale
        self.forearms = (left, right)
        self.forearm_contact = clamp((1.25-wrist_gap)/.75)

    def process(self, frame, now, cv2, np, detect_foam=True):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(now * 1000)
        self._update_forearms(image, timestamp_ms)
        output = self.model.detect_for_video(image, timestamp_ms)
        detected = output.hand_landmarks or []
        points = tuple(tuple((p.x, p.y) for p in hand) for hand in detected)
        if not points:
            self.rubbing.reset()
            self.local_motion.reset()
            return VisionResult(Observation(now, 0, 0, 0), tuple(), 0, 0, 0, 0,
                                self.forearms, self.forearm_contact)

        local_motion = self.local_motion.update(frame, points, cv2, np)
        if len(points) == 1:
            # During genuine palm-to-palm rubbing, one hand frequently occludes
            # the other. Accept that merged view briefly only after two hands
            # have been established, and require strong internal crop motion.
            recently_two = now-self.last_two_hands_at <= 3.0
            arms_confirm = len(self.forearms) == 2 and self.forearm_contact >= .35
            merged_confirmed = recently_two or arms_confirm
            confidence = local_motion * max(.7, self.forearm_contact) if merged_confirmed else 0.0
            return VisionResult(Observation(now, .9 if merged_confirmed else .5,
                                            confidence, 0), points,
                                max(self.forearm_contact, .8 if recently_two else 0.0),
                                local_motion, 0, 0, self.forearms,
                                self.forearm_contact)

        # Stabilize ordering because Tasks may return hands in a different order.
        points = tuple(sorted(points, key=lambda hand: palm_center(hand)[0]))
        self.last_two_hands_at = now
        rub, contact, motion, reversals = self.rubbing.update(*points)
        # Local appearance motion complements landmarks when fingers occlude.
        rub = max(rub, contact * local_motion * .9)
        motion = max(motion, local_motion)
        soap = self.foam.update(frame, points, cv2, np) if detect_foam else 0.0
        return VisionResult(Observation(now, .98, rub, soap), points,
                            contact, motion, reversals, soap, self.forearms,
                            self.forearm_contact)
