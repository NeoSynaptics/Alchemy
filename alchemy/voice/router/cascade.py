"""Cascade strategies — confidence-based escalation between models.

Today: 14B → 72B when the 14B says "I can't access your screen".
Tomorrow: pluggable strategies for quality-based cascades, draft-verify, etc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from alchemy.voice.models.schemas import ModelLocation, RouteDecision, RouteIntent


class CascadeStrategy(ABC):
    """Base class for cascade/escalation strategies."""

    @abstractmethod
    def should_escalate(self, decision: RouteDecision, response_text: str) -> bool:
        """Decide whether to escalate from current model to a stronger one."""

    @abstractmethod
    def escalation_target(self, current_decision: RouteDecision) -> RouteDecision:
        """Return the decision for the escalation target."""


class ConversationToVisionCascade(CascadeStrategy):
    """Escalate from 14B to 72B when the 14B admits it can't do GUI work.

    Catches cases where the keyword pre-filter missed a GUI task and the
    14B didn't tag it properly, but the response content reveals a GUI need.
    """

    _ESCALATION_SIGNALS = [
        "i can't access",
        "i don't have the ability to",
        "i cannot interact with",
        "i'm unable to open",
        "i can't click",
        "i would need to use",
        "i don't have access to your screen",
        "i can't see your screen",
        "i cannot browse",
        "i can't open",
    ]

    def should_escalate(self, decision: RouteDecision, response_text: str) -> bool:
        if decision.intent == RouteIntent.GUI_TASK:
            return False  # Already going to vision
        if not decision.escalation_possible:
            return False
        lower = response_text.lower()
        return any(signal in lower for signal in self._ESCALATION_SIGNALS)

    def escalation_target(self, current_decision: RouteDecision) -> RouteDecision:
        return RouteDecision(
            intent=RouteIntent.GUI_TASK,
            target_model="ui-tars:72b",
            target_location=ModelLocation.CPU_REMOTE,
            confidence=0.75,
            reasoning="Escalated: 14B response indicates GUI capability needed",
            should_stream=False,
            escalation_possible=False,
        )
