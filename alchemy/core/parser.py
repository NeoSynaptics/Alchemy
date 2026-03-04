"""Parse Playwright agent LLM output into structured actions.

Expected format from the LLM:
    Thought: I see a search box. I should click on it.
    Action: click @e5

    Thought: I need to type the search query.
    Action: type @e5 "pole vault"

    Thought: The page is loading.
    Action: wait

    Thought: Task complete — search results are showing.
    Action: done
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# --- Regex patterns ---
_THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?=\nAction:|\Z)", re.DOTALL)
_ACTION_LINE_RE = re.compile(r"Action:\s*(.+?)$", re.MULTILINE)
_REF_RE = re.compile(r"@(e\d+)")
_QUOTED_TEXT_RE = re.compile(r'"([^"]*)"')


@dataclass
class PlaywrightAction:
    """Parsed action from the Playwright agent LLM."""

    type: str  # click, type, scroll, key, select, wait, done
    ref: str | None = None  # "e5"
    text: str | None = None  # for type/select
    direction: str | None = None  # scroll up/down
    key_name: str | None = None  # Enter, Tab, etc.
    thought: str = ""


class ParseError(Exception):
    """Raised when LLM output cannot be parsed."""


def parse_playwright_response(raw: str) -> PlaywrightAction:
    """Parse LLM response into a PlaywrightAction.

    Handles formats:
        click @e5
        type @e5 "some text"
        scroll down
        scroll up
        key Enter
        select @e5 "option text"
        wait
        done

    Raises:
        ParseError: If the response cannot be parsed.
    """
    # Extract thought
    thought = ""
    thought_match = _THOUGHT_RE.search(raw)
    if thought_match:
        thought = thought_match.group(1).strip()

    # Extract action line
    action_match = _ACTION_LINE_RE.search(raw)
    if not action_match:
        raise ParseError(f"No 'Action:' line found in: {raw!r}")

    action_line = action_match.group(1).strip()
    parts = action_line.split(None, 1)  # Split on first whitespace

    if not parts:
        raise ParseError(f"Empty action line: {action_line!r}")

    action_type = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    # Parse based on action type
    if action_type in ("done", "wait"):
        return PlaywrightAction(type=action_type, thought=thought)

    if action_type == "scroll":
        direction = rest.lower() if rest else "down"
        if direction not in ("up", "down"):
            direction = "down"
        return PlaywrightAction(type="scroll", direction=direction, thought=thought)

    if action_type == "key":
        key_name = rest.strip() if rest else None
        if not key_name:
            raise ParseError("key action requires a key name")
        return PlaywrightAction(type="key", key_name=key_name, thought=thought)

    if action_type == "click":
        ref = _extract_ref(rest)
        if not ref:
            raise ParseError(f"click requires a ref (@eN): {action_line!r}")
        return PlaywrightAction(type="click", ref=ref, thought=thought)

    if action_type in ("type", "fill"):
        ref = _extract_ref(rest)
        text = _extract_quoted_text(rest)
        if not ref:
            raise ParseError(f"type requires a ref (@eN): {action_line!r}")
        if text is None:
            raise ParseError(f"type requires quoted text: {action_line!r}")
        return PlaywrightAction(type="type", ref=ref, text=text, thought=thought)

    if action_type == "select":
        ref = _extract_ref(rest)
        text = _extract_quoted_text(rest)
        if not ref:
            raise ParseError(f"select requires a ref (@eN): {action_line!r}")
        if text is None:
            raise ParseError(f"select requires quoted text: {action_line!r}")
        return PlaywrightAction(type="select", ref=ref, text=text, thought=thought)

    raise ParseError(f"Unknown action type: {action_type!r}")


def _extract_ref(text: str) -> str | None:
    """Extract ref ID from text (e.g., '@e5' → 'e5')."""
    match = _REF_RE.search(text)
    return match.group(1) if match else None


def _extract_quoted_text(text: str) -> str | None:
    """Extract double-quoted text from action arguments."""
    match = _QUOTED_TEXT_RE.search(text)
    return match.group(1) if match else None
