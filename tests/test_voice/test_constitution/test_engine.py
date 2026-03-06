"""Tests for ConstitutionEngine — full rule evaluation pipeline."""

from alchemy.voice.schemas import ActionTier, VisionAction
from alchemy.voice.constitution.engine import ConstitutionEngine
from alchemy.voice.constitution.rules import (
    ConstitutionalRule,
    DestructiveActionRule,
    FinancialActionRule,
    RuleVerdict,
)


class TestConstitutionEngine:
    def test_default_rules_loaded(self):
        engine = ConstitutionEngine()
        assert len(engine.rules) == 5

    def test_custom_rules(self):
        engine = ConstitutionEngine(rules=[DestructiveActionRule()])
        assert len(engine.rules) == 1

    def test_safe_action_returns_none(self):
        engine = ConstitutionEngine()
        action = VisionAction(action="click", reasoning="Open browser")
        verdict = engine.evaluate(action, "Open Chrome")
        assert verdict is None

    def test_destructive_returns_approve(self):
        engine = ConstitutionEngine()
        action = VisionAction(action="click", reasoning="Delete all files")
        verdict = engine.evaluate(action, "Clean up")
        assert verdict is not None
        assert verdict.tier == ActionTier.APPROVE
        assert verdict.rule_name == "destructive_action"

    def test_highest_tier_wins(self):
        engine = ConstitutionEngine()
        # "send money" triggers both financial (APPROVE) and communication (NOTIFY)
        action = VisionAction(action="click", reasoning="Send money to contact")
        verdict = engine.evaluate(action, "Transfer and send")
        assert verdict is not None
        assert verdict.tier == ActionTier.APPROVE

    def test_enforce_escalates_tier(self):
        engine = ConstitutionEngine()
        action = VisionAction(
            action="click",
            reasoning="Delete the database",
            tier=ActionTier.AUTO,
        )
        enforced = engine.enforce(action, "Cleanup")
        assert enforced.tier == ActionTier.APPROVE

    def test_enforce_no_downgrade(self):
        engine = ConstitutionEngine()
        # Already APPROVE, no rule fires → stays APPROVE
        action = VisionAction(
            action="click",
            reasoning="Open browser",
            tier=ActionTier.APPROVE,
        )
        enforced = engine.enforce(action, "Browsing")
        assert enforced.tier == ActionTier.APPROVE

    def test_enforce_passthrough_safe(self):
        engine = ConstitutionEngine()
        action = VisionAction(
            action="click",
            reasoning="Open Notepad",
            tier=ActionTier.AUTO,
        )
        enforced = engine.enforce(action, "Write notes")
        assert enforced.tier == ActionTier.AUTO

    def test_rule_exception_handled(self):
        class BrokenRule(ConstitutionalRule):
            name = "broken"

            def evaluate(self, action, goal):
                raise ValueError("boom")

        engine = ConstitutionEngine(rules=[BrokenRule(), DestructiveActionRule()])
        action = VisionAction(action="click", reasoning="Delete file")
        # Should still get verdict from the non-broken rule
        verdict = engine.evaluate(action, "Cleanup")
        assert verdict is not None
        assert verdict.tier == ActionTier.APPROVE

    def test_empty_rules_returns_none(self):
        engine = ConstitutionEngine(rules=[])
        action = VisionAction(action="click", reasoning="Delete everything")
        verdict = engine.evaluate(action, "Destroy")
        assert verdict is None
