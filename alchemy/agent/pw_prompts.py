"""System prompt and formatting for the Playwright GUI agent.

The agent reads an accessibility tree snapshot and outputs one action per turn.
Designed for Qwen3 14B with think: true — deep reasoning per step.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a GUI automation agent operating inside AlchemyOS.
You control applications through their accessibility tree.

RULES:
- Output exactly ONE action per turn
- Use the ref labels from the snapshot (e.g., @e5)
- If unsure, use "scroll down" to see more of the page
- If a page is loading, output "wait"
- If the task is complete, output "done"
- Never guess — if you can't find the element, say so

ACTION FORMAT (use exactly one per response):
  click @REF         — click an element
  type @REF "text"   — type text into an input
  scroll down        — scroll the page down
  scroll up          — scroll the page up
  key KEYNAME        — press a key (Enter, Tab, Escape, etc.)
  select @REF "option" — select from dropdown
  wait               — wait for page to load
  done               — task is complete

RESPONSE FORMAT:
Thought: [brief reasoning about what you see and what to do next]
Action: [exactly one action from the format above]"""


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
    parts.append("Current screen:")
    parts.append(snapshot_text)
    parts.append("")
    parts.append("What is the next action?")

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
