"""Context-aware tier classification — enhances the hardcoded rules.

Considers the task category and goal context when deciding whether
an action should be AUTO, NOTIFY, or APPROVE. Falls back to the
existing classify_tier() for unknown patterns.
"""

from __future__ import annotations

import re

from alchemy.router.categories import TaskCategory
from alchemy.schemas import ActionTier, VisionAction


def classify_tier(action: VisionAction) -> ActionTier:
    """Classify action tier based on action type (base rules).

    Inlined here to avoid importing from alchemy.click (lateral import violation).
    """
    if action.action in ("wait", "done", "fail"):
        return ActionTier.AUTO
    if action.action in ("type", "hotkey"):
        return ActionTier.NOTIFY
    return ActionTier.AUTO

# Keywords in goal or action text that signal destructive intent
_DESTRUCTIVE_KEYWORDS = re.compile(
    r"\b(delete|remove|erase|format|drop|destroy|purge|uninstall|wipe)\b", re.I
)

# Keywords that suggest a "send" action
_SEND_KEYWORDS = re.compile(
    r"\b(send|submit|post|publish|reply|forward|broadcast)\b", re.I
)

# Keywords for purchase / financial actions
_PURCHASE_KEYWORDS = re.compile(
    r"\b(buy|purchase|order|checkout|pay|subscribe|donate)\b", re.I
)


def classify_tier_contextual(
    action: VisionAction,
    category: TaskCategory,
    goal: str,
) -> ActionTier:
    """Context-aware tier classification.

    Uses task category and goal keywords to make smarter decisions
    about which actions need user awareness or approval.
    Falls back to action_parser.classify_tier() for unrecognized patterns.
    """
    # --- Universal dangerous patterns (any category) ---
    if _is_destructive(action, goal):
        return ActionTier.APPROVE

    if _is_purchase(action, goal):
        return ActionTier.APPROVE

    # --- Category-specific rules ---

    if category == TaskCategory.COMMUNICATION:
        # Typing in communication apps should always notify
        if action.action == "type":
            return ActionTier.NOTIFY
        # Clicking near send/submit buttons needs approval
        if action.action == "click" and _goal_implies_send(goal):
            return ActionTier.NOTIFY
        # Hotkeys that send (Enter in some apps, Ctrl+Enter)
        if action.action == "hotkey" and _is_send_hotkey(action.text):
            return ActionTier.APPROVE

    if category == TaskCategory.FILE:
        # Destructive hotkeys in file context
        if action.action == "hotkey" and _is_delete_hotkey(action.text):
            return ActionTier.APPROVE

    # --- Fallback to hardcoded rules ---
    return classify_tier(action)


def _is_destructive(action: VisionAction, goal: str) -> bool:
    """Check if the action + goal context suggests something destructive."""
    text = (action.text or "") + " " + (action.reasoning or "")
    return bool(_DESTRUCTIVE_KEYWORDS.search(text))


def _is_purchase(action: VisionAction, goal: str) -> bool:
    """Check if the action context suggests a purchase or payment."""
    text = (action.text or "") + " " + (action.reasoning or "")
    if _PURCHASE_KEYWORDS.search(text):
        return True
    if _PURCHASE_KEYWORDS.search(goal):
        # Only upgrade to APPROVE if the action is a click (confirming purchase)
        return action.action == "click"
    return False


def _goal_implies_send(goal: str) -> bool:
    """Check if the goal involves sending something."""
    return bool(_SEND_KEYWORDS.search(goal))


def _is_send_hotkey(key: str | None) -> bool:
    """Check if a hotkey is commonly used to send messages."""
    if not key:
        return False
    key_lower = key.lower().replace(" ", "")
    return key_lower in ("enter", "ctrl+enter", "ctrl+return")


def _is_delete_hotkey(key: str | None) -> bool:
    """Check if a hotkey is commonly used to delete."""
    if not key:
        return False
    key_lower = key.lower().replace(" ", "")
    return key_lower in ("delete", "ctrl+d", "shift+delete")
