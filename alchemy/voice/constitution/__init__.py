"""Defense Constitution — 3-tier approval gates (AUTO / NOTIFY / APPROVE).

Hardcoded safety rules that evaluate actions before execution.
Rules cannot be overridden by the model — they are the final defense layer.
"""

from alchemy.voice.constitution.engine import ConstitutionEngine
from alchemy.voice.constitution.rules import (
    AuthenticationRule,
    CommunicationRule,
    ConstitutionalRule,
    DestructiveActionRule,
    FinancialActionRule,
    RuleVerdict,
    SystemModificationRule,
)

__all__ = [
    "ConstitutionEngine",
    "ConstitutionalRule",
    "RuleVerdict",
    "DestructiveActionRule",
    "FinancialActionRule",
    "CommunicationRule",
    "AuthenticationRule",
    "SystemModificationRule",
]
