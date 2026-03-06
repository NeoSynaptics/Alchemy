"""Constitution engine — evaluates actions against all rules.

The engine runs all registered rules against an action and returns
the highest-tier verdict. Rules are composable and order-independent.
"""

from __future__ import annotations

import logging

from alchemy.voice.schemas import ActionTier, VisionAction
from alchemy.voice.constitution.rules import (
    AuthenticationRule,
    CommunicationRule,
    ConstitutionalRule,
    DestructiveActionRule,
    FinancialActionRule,
    RuleVerdict,
    SystemModificationRule,
)

logger = logging.getLogger(__name__)

# Tier priority: APPROVE > NOTIFY > AUTO
_TIER_PRIORITY = {
    ActionTier.AUTO: 0,
    ActionTier.NOTIFY: 1,
    ActionTier.APPROVE: 2,
}


def _build_default_rules() -> list[ConstitutionalRule]:
    """Create the standard rule set."""
    return [
        DestructiveActionRule(),
        FinancialActionRule(),
        CommunicationRule(),
        AuthenticationRule(),
        SystemModificationRule(),
    ]


class ConstitutionEngine:
    """Evaluates actions against constitutional rules.

    Returns the highest-tier verdict across all rules. If no rule fires,
    the action's original tier is preserved (pass-through).
    """

    def __init__(self, rules: list[ConstitutionalRule] | None = None) -> None:
        self._rules = rules if rules is not None else _build_default_rules()

    @property
    def rules(self) -> list[ConstitutionalRule]:
        return list(self._rules)

    def evaluate(self, action: VisionAction, goal: str = "") -> RuleVerdict | None:
        """Run all rules. Return the highest-tier verdict, or None if all pass.

        The returned verdict represents the minimum required tier. If it is
        higher than the action's current tier, the action should be escalated.
        """
        highest: RuleVerdict | None = None
        highest_priority = -1

        for rule in self._rules:
            try:
                verdict = rule.evaluate(action, goal)
            except Exception:
                logger.exception("Rule %s raised an exception", rule.name)
                continue

            if verdict is None:
                continue

            priority = _TIER_PRIORITY.get(verdict.tier, 0)
            if priority > highest_priority:
                highest = verdict
                highest_priority = priority

            # Short-circuit: can't go higher than APPROVE
            if verdict.tier == ActionTier.APPROVE:
                break

        return highest

    def enforce(self, action: VisionAction, goal: str = "") -> VisionAction:
        """Evaluate and return a copy of the action with the enforced tier.

        If constitutional rules require a higher tier than the action already
        has, the tier is escalated. Never downgrades.
        """
        verdict = self.evaluate(action, goal)

        if verdict is None:
            return action

        current_priority = _TIER_PRIORITY.get(action.tier, 0)
        verdict_priority = _TIER_PRIORITY.get(verdict.tier, 0)

        if verdict_priority > current_priority:
            logger.info(
                "Constitution escalated %s → %s (%s: %s)",
                action.tier.value,
                verdict.tier.value,
                verdict.rule_name,
                verdict.reason,
            )
            return action.model_copy(update={"tier": verdict.tier})

        return action
