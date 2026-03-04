"""Approval gate — detect irreversible actions and pause for human confirmation.

The agent does ALL prep work autonomously. It only pauses at the point of no return:
sending emails, deleting files, making purchases, submitting forms.

The gate checks the action + surrounding context (element labels, page URL)
to decide if an action is irreversible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from alchemy.core.parser import PlaywrightAction

logger = logging.getLogger(__name__)

# Keywords in element names or page context that signal irreversible actions
_IRREVERSIBLE_KEYWORDS = frozenset({
    "send", "submit", "delete", "remove", "purchase", "buy", "pay",
    "confirm", "checkout", "place order", "transfer", "publish",
    "unsubscribe", "cancel subscription", "close account",
    "permanently", "irreversible",
})

# Keywords that are safe even if they match partial words
_SAFE_OVERRIDES = frozenset({
    "search", "filter", "sort", "navigate", "back", "cancel",
    "close", "dismiss", "undo",
})

# Roles that are more likely to be irreversible when clicked
_HIGH_RISK_ROLES = frozenset({"button", "link"})


@dataclass
class ApprovalGate:
    """Checks whether an action needs human approval before execution.

    Args:
        enabled: Whether the gate is active (user can toggle in settings).
        extra_keywords: Additional keywords to treat as irreversible.
    """

    enabled: bool = True
    extra_keywords: set[str] = field(default_factory=set)

    def needs_approval(
        self,
        action: PlaywrightAction,
        element_name: str = "",
        page_url: str = "",
    ) -> bool:
        """Check if an action requires human approval.

        Args:
            action: The parsed action about to be executed.
            element_name: The label/name of the target element.
            page_url: Current page URL for context.

        Returns:
            True if the action should pause for human confirmation.
        """
        if not self.enabled:
            return False

        # Only click, type, select, key can be irreversible
        if action.type in ("scroll", "wait", "done"):
            return False

        # Check element name for irreversible keywords
        check_text = f"{element_name} {action.text or ''} {action.thought}".lower()

        # Quick safe override check
        for safe in _SAFE_OVERRIDES:
            if safe in check_text:
                return False

        all_keywords = _IRREVERSIBLE_KEYWORDS | self.extra_keywords
        for keyword in all_keywords:
            if keyword in check_text:
                logger.info(
                    "Approval required: keyword '%s' found in '%s'",
                    keyword, check_text[:100],
                )
                return True

        return False


def is_irreversible(action: PlaywrightAction, element_name: str = "") -> bool:
    """Quick check — stateless version of ApprovalGate.needs_approval."""
    gate = ApprovalGate(enabled=True)
    return gate.needs_approval(action, element_name)
