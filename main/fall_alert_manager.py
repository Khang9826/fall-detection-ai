import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class _PersonAlertState:
    last_state: str = "NORMAL"
    state_count: int = 0
    fall_confirm_count: int = 0
    risk_confirm_count: int = 0
    resolved_count: int = 0
    last_alert_ts: float = 0.0
    alert_active: bool = False
    last_confidence: float = 0.0
    last_bbox: Optional[Tuple[int, int, int, int]] = None


class FallAlertManager:
    """
    Tracks per-person FSM states and emits structured alert events.
    - Multi-frame confirmation to reduce false positives.
    - Cooldown to avoid alert spam.
    - Resolution when state returns to NORMAL for enough frames.
    """

    def __init__(
        self,
        fall_confirm_frames: int = 3,
        risk_confirm_frames: int = 5,
        resolve_frames: int = 5,
        cooldown_sec: float = 10.0,
        min_confidence: float = 0.6
    ):
        self.fall_confirm_frames = fall_confirm_frames
        self.risk_confirm_frames = risk_confirm_frames
        self.resolve_frames = resolve_frames
        self.cooldown_sec = cooldown_sec
        self.min_confidence = min_confidence
        self._people: Dict[int, _PersonAlertState] = {}

    def update(
        self,
        person_id: int,
        state: str,
        confidence: float,
        bbox: Tuple[int, int, int, int],
        timestamp: Optional[float] = None
    ) -> List[dict]:
        """
        Returns a list of alert events for this person on this frame.
        """
        now = timestamp if timestamp is not None else time.time()
        ps = self._people.get(person_id)
        if ps is None:
            ps = _PersonAlertState()
            self._people[person_id] = ps

        ps.last_confidence = confidence
        ps.last_bbox = bbox

        # Track consecutive state frames
        if state == ps.last_state:
            ps.state_count += 1
        else:
            ps.state_count = 1
            ps.last_state = state

        events: List[dict] = []

        # Resolution logic (return to NORMAL for N frames)
        if state == "NORMAL":
            ps.resolved_count += 1
        else:
            ps.resolved_count = 0

        if ps.alert_active and ps.resolved_count >= self.resolve_frames:
            ps.alert_active = False
            events.append(self._make_event(
                person_id=person_id,
                state="NORMAL",
                severity="resolved",
                confidence=confidence,
                timestamp=now,
                bbox=bbox,
                event_type="alert_resolved"
            ))

        # Confidence gate
        if confidence < self.min_confidence:
            return events

        # Cooldown gate
        in_cooldown = (now - ps.last_alert_ts) < self.cooldown_sec

        # RISK warning logic (non-critical)
        if state == "RISK":
            ps.risk_confirm_count += 1
            if (ps.risk_confirm_count >= self.risk_confirm_frames) and not in_cooldown:
                events.append(self._make_event(
                    person_id=person_id,
                    state="RISK",
                    severity="warning",
                    confidence=confidence,
                    timestamp=now,
                    bbox=bbox,
                    event_type="risk_warning"
                ))
                ps.last_alert_ts = now
        else:
            ps.risk_confirm_count = 0

        # FALL alert logic
        if state == "FALL":
            ps.fall_confirm_count += 1
            if (ps.fall_confirm_count >= self.fall_confirm_frames) and not in_cooldown:
                events.append(self._make_event(
                    person_id=person_id,
                    state="FALL",
                    severity="critical",
                    confidence=confidence,
                    timestamp=now,
                    bbox=bbox,
                    event_type="fall_alert"
                ))
                ps.alert_active = True
                ps.last_alert_ts = now
        else:
            ps.fall_confirm_count = 0

        return events

    @staticmethod
    def _make_event(
        person_id: int,
        state: str,
        severity: str,
        confidence: float,
        timestamp: float,
        bbox: Tuple[int, int, int, int],
        event_type: str
    ) -> dict:
        return {
            "event_type": event_type,
            "person_id": int(person_id),
            "state": state,
            "severity": severity,
            "confidence": float(confidence),
            "timestamp": float(timestamp),
            "bbox": [int(b) for b in bbox]
        }


def emit_alerts(socketio, events: List[dict]) -> None:
    """
    Socket.IO integration: emits fall_alert, risk_warning, alert_resolved events.
    """
    for ev in events:
        if ev["event_type"] == "fall_alert":
            socketio.emit("fall_alert", ev)
        elif ev["event_type"] == "risk_warning":
            socketio.emit("risk_warning", ev)
        elif ev["event_type"] == "alert_resolved":
            socketio.emit("alert_resolved", ev)
