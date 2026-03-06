"""AlchemyBrowser — Playwright path for web/Electron automation.

Uses accessibility tree + ref-based actions via Qwen3 14B.
No screenshots, no pixel coordinates. Pure structured data.

Covers: Chrome, Electron apps (VS Code, Spotify, Slack, Discord, Notion).
"""

from alchemy.core.agent import PlaywrightAgent, AgentResult, AgentStatus, StepResult
from alchemy.core.parser import (
    ParseError,
    PlaywrightAction,
    parse_playwright_response,
)
from alchemy.core.prompts import (
    SYSTEM_PROMPT as BROWSER_SYSTEM_PROMPT,
    format_action_log_entry,
    format_user_prompt,
)

__all__ = [
    "PlaywrightAgent",
    "AgentResult",
    "AgentStatus",
    "StepResult",
    "ParseError",
    "PlaywrightAction",
    "parse_playwright_response",
    "BROWSER_SYSTEM_PROMPT",
    "format_action_log_entry",
    "format_user_prompt",
]
