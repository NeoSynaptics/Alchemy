"""Gate module — Claude Code tool call reviewer.

Second-class module that borrows inference from core's shared Ollama client.
Static policies handle safe/dangerous calls instantly; ambiguous calls
go to Qwen3 14B (think:false) for sub-second review.
"""

from alchemy.gate.policies import PolicyDecision, check_static_policy
from alchemy.gate.reviewer import GateResult, GateReviewer

__all__ = [
    "PolicyDecision",
    "check_static_policy",
    "GateResult",
    "GateReviewer",
]
