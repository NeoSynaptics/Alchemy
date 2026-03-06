"""Tests for CascadeStrategy — escalation signal detection."""

from alchemy.voice.models.schemas import ModelLocation, RouteDecision, RouteIntent
from alchemy.voice.router.cascade import ConversationToVisionCascade


def _conversation_decision(escalation_possible: bool = True) -> RouteDecision:
    return RouteDecision(
        intent=RouteIntent.CONVERSATION,
        target_model="qwen3:14b",
        target_location=ModelLocation.GPU_LOCAL,
        confidence=0.8,
        reasoning="test",
        escalation_possible=escalation_possible,
    )


class TestConversationToVisionCascade:
    def setup_method(self):
        self.cascade = ConversationToVisionCascade()

    def test_escalates_on_cant_access(self):
        decision = _conversation_decision()
        assert self.cascade.should_escalate(decision, "I can't access your email directly.")

    def test_escalates_on_cant_click(self):
        decision = _conversation_decision()
        assert self.cascade.should_escalate(decision, "I can't click on buttons for you.")

    def test_escalates_on_cant_see_screen(self):
        decision = _conversation_decision()
        assert self.cascade.should_escalate(decision, "I can't see your screen right now.")

    def test_escalates_on_cannot_browse(self):
        decision = _conversation_decision()
        assert self.cascade.should_escalate(decision, "I cannot browse the web for you.")

    def test_no_escalation_for_normal_response(self):
        decision = _conversation_decision()
        assert not self.cascade.should_escalate(decision, "Python is a programming language.")

    def test_no_escalation_when_already_gui(self):
        decision = RouteDecision(
            intent=RouteIntent.GUI_TASK,
            target_model="ui-tars:72b",
            target_location=ModelLocation.CPU_REMOTE,
            confidence=0.9,
            reasoning="test",
        )
        assert not self.cascade.should_escalate(decision, "I can't access your screen.")

    def test_no_escalation_when_not_possible(self):
        decision = _conversation_decision(escalation_possible=False)
        assert not self.cascade.should_escalate(decision, "I can't access your screen.")

    def test_escalation_target(self):
        decision = _conversation_decision()
        target = self.cascade.escalation_target(decision)
        assert target.intent == RouteIntent.GUI_TASK
        assert target.target_model == "ui-tars:72b"
        assert target.target_location == ModelLocation.CPU_REMOTE
        assert target.escalation_possible is False
