"""Intent classifier — keyword pre-filter + 14B response tag parsing.

Two layers:
1. classify_from_keywords() — zero cost, catches obvious cases (<1ms)
2. parse_intent_tag() — extracts [TAG] from 14B model output (zero marginal cost)
"""

from __future__ import annotations

import re

from alchemy.voice.models.schemas import RouteIntent

# System prompt suffix that instructs the 14B to tag its responses
CLASSIFIER_INSTRUCTION = (
    "\n\nIMPORTANT: Start every response with exactly one of these tags on its own line:\n"
    "[CONVERSATION] — for general chat, questions, math, facts, advice\n"
    "[GUI_TASK] — for tasks requiring a computer (open app, click, browse, fill form, send email)\n"
    "[SYSTEM] — for commands about yourself (pause, settings, status)\n"
    "Then continue with your natural response on the next line."
)

_TAG_PATTERN = re.compile(
    r"^\s*\[(CONVERSATION|GUI_TASK|SYSTEM)\]\s*\n?",
    re.IGNORECASE,
)

_TAG_TO_INTENT: dict[str, RouteIntent] = {
    "CONVERSATION": RouteIntent.CONVERSATION,
    "GUI_TASK": RouteIntent.GUI_TASK,
    "SYSTEM": RouteIntent.SYSTEM_COMMAND,
}

# --- Keyword lists (conservative — only unambiguous signals) ---

_GUI_SIGNALS = [
    "open ", "click ", "go to ", "navigate to ", "fill in ", "fill out ",
    "send email", "send the email", "browse to ", "search for ",
    "download ", "upload ", "install ", "type in ", "log in",
    "sign in", "submit ", "book ", "order ", "purchase ",
    "screenshot", "take a screenshot",
]

_SYSTEM_SIGNALS = [
    "pause", "stop listening", "settings", "status", "shut down", "restart",
]


def parse_intent_tag(text: str) -> tuple[RouteIntent, str]:
    """Extract intent tag from the beginning of a model response.

    Returns (intent, cleaned_text) where cleaned_text has the tag removed.
    If no tag found, returns (UNCLEAR, original_text).
    """
    match = _TAG_PATTERN.match(text)
    if match:
        tag = match.group(1).upper()
        intent = _TAG_TO_INTENT.get(tag, RouteIntent.UNCLEAR)
        cleaned = text[match.end():]
        return intent, cleaned
    return RouteIntent.UNCLEAR, text


def classify_from_keywords(text: str) -> RouteIntent | None:
    """Fast pre-check using keyword heuristics before model inference.

    Returns None if uncertain (let the model decide).
    Returns a definite intent only for very clear cases.
    """
    lower = text.lower().strip()

    for signal in _GUI_SIGNALS:
        if signal in lower:
            return RouteIntent.GUI_TASK

    for signal in _SYSTEM_SIGNALS:
        if lower.startswith(signal) or lower == signal:
            return RouteIntent.SYSTEM_COMMAND

    return None
