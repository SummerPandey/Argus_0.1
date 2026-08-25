from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum


class Stage(str, Enum):
    IDLE = "Waiting for hands"
    READY = "Hands detected"
    SOAP = "Soap detected"
    RUBBING = "Keep rubbing"
    COMPLETE = "Handwash complete"
    FAILED = "Session incomplete"


@dataclass(frozen=True)
class Observation:
    timestamp: float
    hands_confidence: float = 0.0
    rubbing_confidence: float = 0.0
    soap_confidence: float = 0.0


@dataclass(frozen=True)
class Snapshot:
    stage: Stage
    hands_present: bool
    soap_seen: bool
    rubbing_seconds: float
    progress: float
    message: str


class SessionEngine:
    """Temporal state machine independent of cameras, models, and UI."""

    def __init__(
        self,
        required_rub_seconds: float = 20.0,
        confirmation_seconds: float = 2.0,
        abandonment_seconds: float = 5.0,
        require_soap: bool = True,
    ):
        self.required_rub_seconds = required_rub_seconds
        self.confirmation_seconds = confirmation_seconds
        self.abandonment_seconds = abandonment_seconds
        self.require_soap = require_soap
        self.reset()

    def reset(self) -> None:
        self.stage = Stage.IDLE
        self.soap_seen = False
        self.rubbing_seconds = 0.0
        self._confirmed = False
        self._candidate_seconds = 0.0
        self._last_time: float | None = None
        self._hands_lost_at: float | None = None
        self._rub_history: deque[float] = deque(maxlen=7)
        self._hand_history: deque[float] = deque(maxlen=5)

    def update(self, observation: Observation) -> Snapshot:
        now = observation.timestamp
        dt = 0.0 if self._last_time is None else max(0.0, min(now - self._last_time, 0.25))
        self._last_time = now

        self._rub_history.append(observation.rubbing_confidence)
        self._hand_history.append(observation.hands_confidence)
        rubbing = sum(self._rub_history) / len(self._rub_history) >= 0.58
        hands = sum(self._hand_history) / len(self._hand_history) >= 0.55

        if observation.soap_confidence >= 0.70:
            self.soap_seen = True

        if self.stage in (Stage.COMPLETE, Stage.FAILED):
            return self.snapshot(hands)

        if hands:
            self._hands_lost_at = None
            if self.stage == Stage.IDLE:
                self.stage = Stage.READY
        elif self.stage != Stage.IDLE:
            if self._hands_lost_at is None:
                self._hands_lost_at = now
            elif now - self._hands_lost_at >= self.abandonment_seconds:
                self.stage = Stage.FAILED
                return self.snapshot(False)

        if self.require_soap and self.soap_seen and self.stage in (Stage.IDLE, Stage.READY):
            self.stage = Stage.SOAP

        if hands and rubbing:
            if not self._confirmed:
                self._candidate_seconds += dt
                if self._candidate_seconds >= self.confirmation_seconds:
                    self._confirmed = True
                    self.rubbing_seconds += self._candidate_seconds
                    self.stage = Stage.RUBBING
            else:
                self.rubbing_seconds += dt
                self.stage = Stage.RUBBING
        elif not self._confirmed:
            self._candidate_seconds = 0.0

        if self.rubbing_seconds >= self.required_rub_seconds:
            self.rubbing_seconds = self.required_rub_seconds
            self.stage = Stage.COMPLETE if (self.soap_seen or not self.require_soap) else Stage.FAILED

        return self.snapshot(hands)

    def snapshot(self, hands_present: bool | None = None) -> Snapshot:
        hands = bool(self._hand_history and sum(self._hand_history) / len(self._hand_history) >= 0.55)
        if hands_present is not None:
            hands = hands_present
        messages = {
            Stage.IDLE: "Place both hands in view",
            Stage.READY: ("Apply soap, then rub your hands" if self.require_soap
                          else "Room test — rub your hands together"),
            Stage.SOAP: "Great — rub palms together",
            Stage.RUBBING: "Nice work — keep rubbing",
            Stage.COMPLETE: "Sparkling clean!",
            Stage.FAILED: "Let’s try that wash again",
        }
        return Snapshot(
            stage=self.stage,
            hands_present=hands,
            soap_seen=self.soap_seen,
            rubbing_seconds=self.rubbing_seconds,
            progress=min(1.0, self.rubbing_seconds / self.required_rub_seconds),
            message=messages[self.stage],
        )
