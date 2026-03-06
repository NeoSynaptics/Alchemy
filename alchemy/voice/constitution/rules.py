"""Constitutional rules — hardcoded safety checks for action approval.

Rules run BEFORE Alchemy executes anything. They evaluate actions from
the vision agent and determine the minimum required tier (AUTO/NOTIFY/APPROVE).
Rules cannot be overridden by the model — they are the final defense layer.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from alchemy.voice.schemas import ActionTier, VisionAction

logger = logging.getLogger(__name__)


@dataclass
class RuleVerdict:
    """Result of a constitutional rule evaluation."""

    tier: ActionTier
    rule_name: str
    reason: str


class ConstitutionalRule(ABC):
    """Base class for a constitutional safety rule."""

    name: str = "unnamed"

    @abstractmethod
    def evaluate(self, action: VisionAction, goal: str) -> RuleVerdict | None:
        """Evaluate an action. Return a verdict to override tier, or None to skip."""


class DestructiveActionRule(ConstitutionalRule):
    """Escalate destructive actions to APPROVE tier.

    Catches actions that delete data, close applications, or modify system settings
    regardless of what the model thinks the tier should be.
    """

    name = "destructive_action"

    _DESTRUCTIVE_PATTERNS = [
        r"\bdelete\b",
        r"\bremove\b",
        r"\buninstall\b",
        r"\bformat\b",
        r"\berase\b",
        r"\bshutdown\b",
        r"\brestart\b",
        r"\bclose all\b",
        r"\bterminate\b",
        r"\bkill\b",
        r"\bdrop\b",
        r"\bwipe\b",
    ]

    _COMPILED = [re.compile(p, re.IGNORECASE) for p in _DESTRUCTIVE_PATTERNS]

    def evaluate(self, action: VisionAction, goal: str) -> RuleVerdict | None:
        text_to_check = f"{action.reasoning} {action.text or ''} {goal}"
        for pattern in self._COMPILED:
            if pattern.search(text_to_check):
                return RuleVerdict(
                    tier=ActionTier.APPROVE,
                    rule_name=self.name,
                    reason=f"Destructive keyword detected: {pattern.pattern}",
                )
        return None


class FinancialActionRule(ConstitutionalRule):
    """Escalate financial actions (purchases, payments, transfers) to APPROVE.

    Any action involving money or payment must be explicitly approved.
    """

    name = "financial_action"

    _FINANCIAL_PATTERNS = [
        r"\b(?:buy|purchase|order|checkout|pay|payment)\b",
        r"\b(?:transfer|send money|wire|deposit|withdraw)\b",
        r"\b(?:subscribe|upgrade plan|billing)\b",
        r"\$\d+",
        r"\b\d+\.\d{2}\b",  # Price-like patterns (19.99)
    ]

    _COMPILED = [re.compile(p, re.IGNORECASE) for p in _FINANCIAL_PATTERNS]

    def evaluate(self, action: VisionAction, goal: str) -> RuleVerdict | None:
        text_to_check = f"{action.reasoning} {action.text or ''} {goal}"
        for pattern in self._COMPILED:
            if pattern.search(text_to_check):
                return RuleVerdict(
                    tier=ActionTier.APPROVE,
                    rule_name=self.name,
                    reason=f"Financial action detected: {pattern.pattern}",
                )
        return None


class CommunicationRule(ConstitutionalRule):
    """Escalate outbound communications to NOTIFY or APPROVE.

    Sending messages, emails, or posts that leave the system should be visible
    to the user. Sending to unknown recipients requires approval.
    """

    name = "communication"

    _SEND_PATTERNS = [
        r"\b(?:send|post|publish|submit|tweet|reply|forward)\b",
    ]

    _HIGH_RISK_PATTERNS = [
        r"\b(?:send to|email to|message to)\b",
        r"\b(?:all contacts|broadcast|mass)\b",
    ]

    _SEND_COMPILED = [re.compile(p, re.IGNORECASE) for p in _SEND_PATTERNS]
    _HIGH_COMPILED = [re.compile(p, re.IGNORECASE) for p in _HIGH_RISK_PATTERNS]

    def evaluate(self, action: VisionAction, goal: str) -> RuleVerdict | None:
        text_to_check = f"{action.reasoning} {action.text or ''} {goal}"

        # Check high-risk first
        for pattern in self._HIGH_COMPILED:
            if pattern.search(text_to_check):
                return RuleVerdict(
                    tier=ActionTier.APPROVE,
                    rule_name=self.name,
                    reason=f"High-risk communication: {pattern.pattern}",
                )

        # Standard send → NOTIFY
        for pattern in self._SEND_COMPILED:
            if pattern.search(text_to_check):
                return RuleVerdict(
                    tier=ActionTier.NOTIFY,
                    rule_name=self.name,
                    reason=f"Outbound communication: {pattern.pattern}",
                )

        return None


class AuthenticationRule(ConstitutionalRule):
    """Escalate login/credential actions to APPROVE.

    Typing passwords, entering credentials, or changing auth settings
    must be explicitly approved.
    """

    name = "authentication"

    _AUTH_PATTERNS = [
        r"\b(?:password|passwd|credential|secret|token|api.?key)\b",
        r"\b(?:sign.?in|log.?in|authenticate)\b",
        r"\b(?:change password|reset password|two.?factor|2fa|mfa)\b",
    ]

    _COMPILED = [re.compile(p, re.IGNORECASE) for p in _AUTH_PATTERNS]

    def evaluate(self, action: VisionAction, goal: str) -> RuleVerdict | None:
        # Only trigger on type actions (entering credentials)
        if action.action != "type":
            return None

        text_to_check = f"{action.reasoning} {action.text or ''} {goal}"
        for pattern in self._COMPILED:
            if pattern.search(text_to_check):
                return RuleVerdict(
                    tier=ActionTier.APPROVE,
                    rule_name=self.name,
                    reason=f"Credential entry detected: {pattern.pattern}",
                )
        return None


class SystemModificationRule(ConstitutionalRule):
    """Escalate system-level modifications to APPROVE.

    Registry edits, driver changes, firewall rules, etc.
    """

    name = "system_modification"

    _SYSTEM_PATTERNS = [
        r"\b(?:registry|regedit|group.?policy)\b",
        r"\b(?:firewall|security.?settings|permissions)\b",
        r"\b(?:driver|service|daemon)\b",
        r"\b(?:environment.?variable|PATH|system.?variable)\b",
        r"\b(?:admin|administrator|sudo|root)\b",
    ]

    _COMPILED = [re.compile(p, re.IGNORECASE) for p in _SYSTEM_PATTERNS]

    def evaluate(self, action: VisionAction, goal: str) -> RuleVerdict | None:
        text_to_check = f"{action.reasoning} {action.text or ''} {goal}"
        for pattern in self._COMPILED:
            if pattern.search(text_to_check):
                return RuleVerdict(
                    tier=ActionTier.APPROVE,
                    rule_name=self.name,
                    reason=f"System modification: {pattern.pattern}",
                )
        return None
