"""Parse UI-TARS model output into structured VisionAction objects.

UI-TARS outputs text like:
    Thought: I see a search box in the top-right corner.
    Action: click(start_box='(452,128)')

This module extracts the thought and action, parses coordinates,
and converts to VisionAction with pixel coordinates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from alchemy.schemas import ActionTier, VisionAction

# --- Regex patterns ---
_THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?=\nAction:|\Z)", re.DOTALL)
_ACTION_RE = re.compile(r"Action:\s*(\w+)\((.*)\)\s*$", re.MULTILINE)
_COORD_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
_CONTENT_RE = re.compile(r"content='((?:[^'\\]|\\.)*)'")
_KEY_RE = re.compile(r"key='([^']*)'")
_DIRECTION_RE = re.compile(r"direction='([^']*)'")
_AMOUNT_RE = re.compile(r"amount=(\d+)")
# Handles both formats:
#   start_box='(452,128)'                          — 7B / simple format
#   start_box='<|box_start|>(13,980)<|box_end|>'   — 72B-DPO native format
_START_BOX_RE = re.compile(r"start_box='([^']+)'")
_END_BOX_RE = re.compile(r"end_box='([^']+)'")

# Map UI-TARS action names to VisionAction.action values
_ACTION_MAP = {
    "click": "click",
    "left_double": "double_click",
    "right_single": "right_click",
    "drag": "drag",
    "type": "type",
    "hotkey": "hotkey",
    "scroll": "scroll",
    "wait": "wait",
    "finished": "done",
}


@dataclass
class ParsedAction:
    """Intermediate parsed representation from UI-TARS raw output."""
    thought: str
    action_type: str
    start_box: tuple[int, int] | None = None
    end_box: tuple[int, int] | None = None
    content: str | None = None
    key: str | None = None
    direction: str | None = None
    amount: int | None = None


def parse_uitars_response(raw: str) -> ParsedAction:
    """Extract Thought and Action from UI-TARS raw output.

    Args:
        raw: Raw text response from the model.

    Returns:
        ParsedAction with extracted fields.

    Raises:
        ValueError: If the response cannot be parsed.
    """
    # Extract thought
    thought_match = _THOUGHT_RE.search(raw)
    thought = thought_match.group(1).strip() if thought_match else ""

    # Extract action
    action_match = _ACTION_RE.search(raw)
    if not action_match:
        raise ValueError(f"Could not parse action from: {raw!r}")

    action_type = action_match.group(1)
    args_str = action_match.group(2)

    # Parse coordinates from start_box
    start_box = None
    start_match = _START_BOX_RE.search(args_str)
    if start_match:
        coord_match = _COORD_RE.search(start_match.group(1))
        if coord_match:
            start_box = (int(coord_match.group(1)), int(coord_match.group(2)))

    # Parse end_box (for drag)
    end_box = None
    end_match = _END_BOX_RE.search(args_str)
    if end_match:
        coord_match = _COORD_RE.search(end_match.group(1))
        if coord_match:
            end_box = (int(coord_match.group(1)), int(coord_match.group(2)))

    # Parse content (for type, finished)
    content = None
    content_match = _CONTENT_RE.search(args_str)
    if content_match:
        content = content_match.group(1).replace("\\'", "'")

    # Parse key (for hotkey)
    key = None
    key_match = _KEY_RE.search(args_str)
    if key_match:
        key = key_match.group(1)

    # Parse direction (for scroll)
    direction = None
    dir_match = _DIRECTION_RE.search(args_str)
    if dir_match:
        direction = dir_match.group(1)

    # Parse amount (for scroll)
    amount = None
    amount_match = _AMOUNT_RE.search(args_str)
    if amount_match:
        amount = int(amount_match.group(1))

    return ParsedAction(
        thought=thought,
        action_type=action_type,
        start_box=start_box,
        end_box=end_box,
        content=content,
        key=key,
        direction=direction,
        amount=amount,
    )


def scale_coord(norm_x: int, norm_y: int, width: int, height: int) -> tuple[int, int]:
    """Convert normalized (0-1000) coordinates to pixel coordinates."""
    px = round(norm_x / 1000 * width)
    py = round(norm_y / 1000 * height)
    return (min(px, width), min(py, height))


def to_vision_action(
    parsed: ParsedAction,
    screen_width: int = 1920,
    screen_height: int = 1080,
) -> VisionAction:
    """Convert ParsedAction to VisionAction with pixel coordinates."""
    action_name = _ACTION_MAP.get(parsed.action_type, parsed.action_type)

    x, y = None, None
    if parsed.start_box:
        x, y = scale_coord(*parsed.start_box, screen_width, screen_height)

    end_x, end_y = None, None
    if parsed.end_box:
        end_x, end_y = scale_coord(*parsed.end_box, screen_width, screen_height)

    text = parsed.content or parsed.key

    return VisionAction(
        action=action_name,
        x=x,
        y=y,
        end_x=end_x,
        end_y=end_y,
        text=text,
        reasoning=parsed.thought,
        tier=ActionTier.AUTO,
        direction=parsed.direction,
        amount=parsed.amount,
    )


def classify_tier(action: VisionAction) -> ActionTier:
    """Classify action tier based on action type.

    Simple rules for now — will be refined with a smarter router later.
    """
    if action.action in ("wait", "done", "fail"):
        return ActionTier.AUTO
    if action.action in ("type", "hotkey"):
        return ActionTier.NOTIFY
    if action.action == "scroll":
        return ActionTier.AUTO
    # clicks are AUTO by default
    return ActionTier.AUTO
