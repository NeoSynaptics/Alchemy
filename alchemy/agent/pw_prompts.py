"""System prompt and formatting for the Playwright GUI agent.

The agent reads an accessibility tree snapshot and outputs one action per turn.
Designed for Qwen3 14B with think: true — deep reasoning per step.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a browser automation agent. You see accessibility tree snapshots and output actions.

ACTIONS (pick exactly ONE per turn):
  click @REF         — click an element
  type @REF "text"   — type text into an input
  scroll down        — scroll the page down
  scroll up          — scroll the page up
  key KEYNAME        — press a key (Enter, Tab, Escape, etc.)
  select @REF "option" — select from dropdown
  wait               — wait for page to load
  done               — task is complete

RULES:
- Dismiss cookie consent or popups first
- If the page is blocked (CAPTCHA, error), output done
- If unsure, scroll down to see more

EXAMPLE:
  Task: Search for cats
  Screen: - searchbox "Search" [ref=e4]
          - button "Go" [ref=e5]
  Response:
  Thought: I see a search box. I will type my query.
  Action: type @e4 "cats"

OUTPUT FORMAT — every response MUST end with exactly:
Thought: [one sentence]
Action: [one action]"""


def format_user_prompt(
    task: str,
    snapshot_text: str,
    action_log: list[str],
    step: int,
    max_log_entries: int = 10,
) -> str:
    """Build the user prompt for the LLM.

    Args:
        task: The user's task description.
        snapshot_text: Formatted accessibility tree.
        action_log: List of previous actions taken.
        step: Current step number.
        max_log_entries: Maximum action log entries to include.

    Returns:
        Formatted user prompt string.
    """
    parts = [f"Task: {task}", "", f"Step: {step}"]

    # Include recent action history
    if action_log:
        recent = action_log[-max_log_entries:]
        parts.append("")
        parts.append("Previous actions:")
        for entry in recent:
            parts.append(f"  {entry}")

    parts.append("")
    parts.append("Accessibility tree (interactive elements have [ref=eN]):")
    parts.append(snapshot_text)
    parts.append("")
    parts.append("Respond with Thought: and Action:")

    return "\n".join(parts)


def format_action_log_entry(
    step: int,
    action_type: str,
    ref: str | None = None,
    text: str | None = None,
    success: bool = True,
    error: str | None = None,
) -> str:
    """Format a single action for the action log.

    Examples:
        "Step 1: click @e5 → OK"
        "Step 2: type @e3 \"hello\" → OK"
        "Step 3: scroll down → OK"
        "Step 4: click @e7 → FAILED: Element not found"
    """
    parts = [f"Step {step}: {action_type}"]

    if ref:
        parts.append(f"@{ref}")
    if text:
        parts.append(f'"{text[:50]}"')

    status = "OK" if success else f"FAILED: {error or 'unknown error'}"
    parts.append(f"→ {status}")

    return " ".join(parts)
