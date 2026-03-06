"""AlchemyClick — 10 Proven Patterns.

Each pattern is a tested, reliable capability that AlchemyClick guarantees
on any Windows machine with the required hardware. Patterns are the
contract between AlchemyClick (core) and any app or workflow that uses it.

Patterns are NOT features — they are infrastructure guarantees.
Apps can assume these patterns work. If a pattern breaks, it's a P0 bug.

Patterns belong to one of three scopes:
  - "click"   — shared AlchemyClick behavior (used by both paths)
  - "flow"    — AlchemyFlow (vision/ghost cursor path)
  - "browser" — AlchemyBrowser (Playwright path)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PatternStatus(str, Enum):
    """Readiness level of a pattern."""
    PROVEN = "proven"      # Tested end-to-end, works on production hardware
    BUILT = "built"        # Code complete, needs more real-world validation
    PLANNED = "planned"    # Architecture designed, not yet implemented


@dataclass(frozen=True)
class Pattern:
    """A proven AlchemyClick capability."""
    id: str
    name: str
    description: str
    status: PatternStatus
    components: tuple[str, ...]  # File paths that implement this pattern
    scope: str = "click"         # "click" (shared), "flow", or "browser"


# ---------------------------------------------------------------------------
# AlchemyFlow Patterns (vision/ghost cursor path)
# ---------------------------------------------------------------------------

SCREENSHOT_VLM_CLICK = Pattern(
    id="screenshot-vlm-click",
    name="Screenshot -> VLM -> Coordinate -> Click",
    description=(
        "The fundamental loop. Capture desktop screenshot, send to Qwen2.5-VL 7B, "
        "parse UI-TARS coordinates from model output, execute click at pixel position. "
        "~3s per step on GPU. Works on any visible UI element."
    ),
    status=PatternStatus.PROVEN,
    scope="flow",
    components=(
        "alchemy/click/flow/vision_agent.py",
        "alchemy/click/flow/action_parser.py",
        "alchemy/click/flow/action_executor.py",
    ),
)

GHOST_CURSOR = Pattern(
    id="ghost-cursor",
    name="Ghost Cursor Overlay",
    description=(
        "Orange 24px tkinter dot, always-on-top, click-through (WS_EX_TRANSPARENT). "
        "Smooth ease-out cubic glide so user can follow. Click flash animation "
        "(#FF6600 -> #FFAA00 for 200ms). Parks in bottom-right corner when idle. "
        "Runs in background thread at 60fps -- never blocks the event loop."
    ),
    status=PatternStatus.PROVEN,
    scope="flow",
    components=(
        "alchemy/desktop/cursor.py",
        "alchemy/desktop/controller.py",
    ),
)

SENDINPUT_ISOLATION = Pattern(
    id="sendinput-isolation",
    name="SendInput Mouse Isolation",
    description=(
        "All clicks/scrolls use Win32 SendInput with MOUSEEVENTF_ABSOLUTE. "
        "User's real cursor position is saved (GetCursorPos) before each action "
        "and restored (SetCursorPos) after. User can keep using their mouse "
        "while the agent clicks independently. Zero interference."
    ),
    status=PatternStatus.PROVEN,
    scope="flow",
    components=(
        "alchemy/desktop/controller.py",
    ),
)

DUAL_FORMAT_PARSING = Pattern(
    id="dual-format-parsing",
    name="Dual Coordinate Format Parsing",
    description=(
        "Handles both UI-TARS v1 (normalized 0-1000) and v1.5 (absolute pixel) "
        "coordinate formats. Supports box_start, point=, start_point syntaxes. "
        "smart_resize_dimensions() computes the model's internal image size for "
        "accurate absolute->screen coordinate mapping."
    ),
    status=PatternStatus.PROVEN,
    scope="flow",
    components=(
        "alchemy/click/flow/action_parser.py",
    ),
)

MODEL_ROUTING = Pattern(
    id="model-routing",
    name="Model Routing + Escalation",
    description=(
        "Fast 7B model handles simple tasks (MEDIA/FILE/SYSTEM categories). "
        "Complex tasks go to full model. If fast model fails inference or produces "
        "3 consecutive parse errors, automatically escalates to full model mid-task. "
        "DeBERTa-v3 router classifies task category in <5ms for routing decisions."
    ),
    status=PatternStatus.BUILT,
    scope="flow",
    components=(
        "alchemy/click/flow/vision_agent.py",
        "alchemy/router/categories.py",
    ),
)

OMNIPARSER_PERCEPTION = Pattern(
    id="omniparser-perception",
    name="OmniParser Fast Perception Layer",
    description=(
        "YOLO + OCR element detection runs BEFORE the VLM (~200-500ms for all elements). "
        "Three modes: (1) Fast path — direct element match skips VLM entirely. "
        "(2) Context enrichment — detected element map injected into VLM prompt. "
        "(3) Post-VLM verification — checks if VLM coordinates land on a real element. "
        "Uses Microsoft OmniParser v2 (MIT). ~1-2GB VRAM."
    ),
    status=PatternStatus.BUILT,
    scope="flow",
    components=(
        "alchemy/click/flow/omniparser.py",
        "alchemy/click/flow/vision_agent.py",
        "alchemy/click/flow/flow_agent.py",
    ),
)

ADAPTIVE_TIMEOUTS = Pattern(
    id="adaptive-timeouts",
    name="Adaptive Timeouts per Task Category",
    description=(
        "Each task category gets a tuned timeout: DEVELOPMENT=480s, WEB=240s, "
        "GENERAL=300s, MEDIA/FILE=180s, COMMUNICATION/SYSTEM=150s. "
        "Screenshot intervals adapt per action: 0.3s min after clicks (fast), "
        "2.0s min after waits (slow). Prevents premature timeouts on complex tasks "
        "and wasted time on simple ones."
    ),
    status=PatternStatus.BUILT,
    scope="flow",
    components=(
        "alchemy/click/flow/vision_agent.py",
    ),
)

# ---------------------------------------------------------------------------
# AlchemyBrowser Patterns (Playwright path)
# ---------------------------------------------------------------------------

PLAYWRIGHT_A11Y = Pattern(
    id="playwright-a11y",
    name="Playwright Tier 1 (Accessibility Tree)",
    description=(
        "For web/Electron apps: Playwright captures the accessibility tree, extracts "
        "ref IDs mapped to ARIA role+name. Qwen3 14B reasons over structured data "
        "(no screenshots needed). Actions use deterministic ref-based locators -- "
        "no pixel coordinate guessing. Covers Chrome, VS Code, Spotify, Slack, Discord."
    ),
    status=PatternStatus.PROVEN,
    scope="browser",
    components=(
        "alchemy/core/agent.py",
        "alchemy/core/executor.py",
        "alchemy/core/parser.py",
        "alchemy/core/prompts.py",
    ),
)

# ---------------------------------------------------------------------------
# Shared AlchemyClick Patterns (used by both paths)
# ---------------------------------------------------------------------------

ACTION_TIER_CLASSIFICATION = Pattern(
    id="action-tier-classification",
    name="3-Tier Action Safety Classification",
    description=(
        "Every action gets a safety tier before execution: "
        "AUTO (click/scroll/wait -- execute silently), "
        "NOTIFY (type/hotkey -- execute + notify user via AlchemyVoice tray), "
        "APPROVE (send email/delete/submit -- pause loop, wait for human approval). "
        "Context-aware: same action can be different tiers based on task category."
    ),
    status=PatternStatus.PROVEN,
    scope="click",
    components=(
        "alchemy/click/flow/action_parser.py",
        "alchemy/router/tier.py",
        "alchemy/schemas.py",
    ),
)

TASK_LIFECYCLE = Pattern(
    id="task-lifecycle",
    name="Task Lifecycle + Approval Gates",
    description=(
        "Full task state machine: PENDING -> RUNNING -> WAITING_APPROVAL -> COMPLETED/FAILED/DENIED. "
        "Approval gates use asyncio.Event for zero-poll waiting. AlchemyVoice receives approval requests "
        "with screenshot + action details. Configurable approval timeout (default 60s). "
        "Tasks can be cancelled at any step via cancel_event."
    ),
    status=PatternStatus.PROVEN,
    scope="click",
    components=(
        "alchemy/click/task_manager.py",
        "alchemy/click/flow/vision_agent.py",
        "alchemy/api/vision.py",
    ),
)

MULTI_STEP_EXECUTION = Pattern(
    id="multi-step-execution",
    name="Multi-Step Execution with History Windowing",
    description=(
        "Up to 50 steps per task (configurable). History windowing keeps recent N steps "
        "in context while trimming older ones to save tokens. Adaptive screenshot intervals "
        "based on action type (fast for clicks, slow for waits). Max-steps guard prevents "
        "infinite loops. Consecutive parse error detection with auto-escalation."
    ),
    status=PatternStatus.PROVEN,
    scope="click",
    components=(
        "alchemy/click/flow/vision_agent.py",
    ),
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_PATTERNS: tuple[Pattern, ...] = (
    # AlchemyFlow (vision/ghost cursor)
    SCREENSHOT_VLM_CLICK,
    GHOST_CURSOR,
    SENDINPUT_ISOLATION,
    DUAL_FORMAT_PARSING,
    MODEL_ROUTING,
    OMNIPARSER_PERCEPTION,
    ADAPTIVE_TIMEOUTS,
    # AlchemyBrowser (Playwright)
    PLAYWRIGHT_A11Y,
    # Shared (AlchemyClick)
    ACTION_TIER_CLASSIFICATION,
    TASK_LIFECYCLE,
    MULTI_STEP_EXECUTION,
)

_PATTERN_MAP: dict[str, Pattern] = {p.id: p for p in ALL_PATTERNS}


def get_pattern(pattern_id: str) -> Pattern | None:
    """Look up a pattern by ID."""
    return _PATTERN_MAP.get(pattern_id)


def proven_patterns() -> list[Pattern]:
    """Return only patterns with PROVEN status."""
    return [p for p in ALL_PATTERNS if p.status == PatternStatus.PROVEN]


def flow_patterns() -> list[Pattern]:
    """Return patterns belonging to AlchemyFlow."""
    return [p for p in ALL_PATTERNS if p.scope == "flow"]


def browser_patterns() -> list[Pattern]:
    """Return patterns belonging to AlchemyBrowser."""
    return [p for p in ALL_PATTERNS if p.scope == "browser"]


def pattern_report() -> str:
    """Human-readable summary of all patterns and their status."""
    lines = [
        "AlchemyClick -- 11 Patterns",
        "=" * 44,
        "",
        "  AlchemyFlow (vision + ghost cursor):",
    ]
    for i, p in enumerate(ALL_PATTERNS, 1):
        if p.scope == "flow":
            _append_pattern_line(lines, i, p)

    lines.append("")
    lines.append("  AlchemyBrowser (Playwright):")
    for i, p in enumerate(ALL_PATTERNS, 1):
        if p.scope == "browser":
            _append_pattern_line(lines, i, p)

    lines.append("")
    lines.append("  Shared (AlchemyClick):")
    for i, p in enumerate(ALL_PATTERNS, 1):
        if p.scope == "click":
            _append_pattern_line(lines, i, p)

    proven = sum(1 for p in ALL_PATTERNS if p.status == PatternStatus.PROVEN)
    lines.append(f"\n  {proven}/{len(ALL_PATTERNS)} proven")
    return "\n".join(lines)


def _append_pattern_line(lines: list[str], i: int, p: Pattern) -> None:
    status_mark = {
        PatternStatus.PROVEN: "[OK]",
        PatternStatus.BUILT: "[--]",
        PatternStatus.PLANNED: "[  ]",
    }[p.status]
    lines.append(f"    {i:2d}. {status_mark} {p.name}")
