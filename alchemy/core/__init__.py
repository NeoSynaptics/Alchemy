"""Alchemy Core — the agent kernel.

This is the protected layer. Outer layers (API, research, router, shadow)
MUST only import from this package's public API — never reach into
core.agent, core.parser, etc. directly.

The core contains:
    - Agent loop (PlaywrightAgent)
    - LLM action parsing
    - Accessibility tree snapshot + ref mapping
    - Browser action execution
    - Vision escalation (Tier 1.5 fallback)
    - Approval gate (irreversible action detection)
    - Protocol contracts (LLMProvider, BrowserProvider, ApprovalChecker)
"""

from alchemy.core.agent import PlaywrightAgent, AgentResult, AgentStatus, StepResult
from alchemy.core.approval import ApprovalGate, is_irreversible
from alchemy.core.browser import BrowserManager
from alchemy.core.escalation import (
    EscalationResult,
    StuckDetector,
    StuckReason,
    VisionEscalation,
)
from alchemy.core.executor import ExecutionError, execute_action
from alchemy.core.parser import ParseError, PlaywrightAction, parse_playwright_response
from alchemy.core.prompts import SYSTEM_PROMPT, format_action_log_entry, format_user_prompt
from alchemy.core.protocols import ApprovalChecker, BrowserProvider, LLMProvider
from alchemy.core.snapshot import RefEntry, SnapshotResult, capture_snapshot
from alchemy.core.trace import AgentTrace, TraceEntry

__all__ = [
    # Agent
    "PlaywrightAgent",
    "AgentResult",
    "AgentStatus",
    "StepResult",
    # Approval
    "ApprovalGate",
    "is_irreversible",
    # Browser
    "BrowserManager",
    # Escalation
    "EscalationResult",
    "StuckDetector",
    "StuckReason",
    "VisionEscalation",
    # Executor
    "ExecutionError",
    "execute_action",
    # Parser
    "ParseError",
    "PlaywrightAction",
    "parse_playwright_response",
    # Prompts
    "SYSTEM_PROMPT",
    "format_action_log_entry",
    "format_user_prompt",
    # Protocols
    "ApprovalChecker",
    "BrowserProvider",
    "LLMProvider",
    # Snapshot
    "RefEntry",
    "SnapshotResult",
    "capture_snapshot",
    # Trace
    "AgentTrace",
    "TraceEntry",
]
