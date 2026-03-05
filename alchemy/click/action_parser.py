"""Parse UI-TARS model output into structured VisionAction objects.

Supports both UI-TARS v1 (72B-DPO) and UI-TARS v1.5 (7B) output formats:

v1 (72B-DPO) — normalized 0-1000 coordinates:
    Thought: I see a search box.
    Action: click(start_box='<|box_start|>(452,128)<|box_end|>')

v1.5 (7B) — absolute pixel coordinates in smart-resized image space:
    Thought: I see a search box.
    Action: click(start_box='<|box_start|>(580,92)<|box_end|>')

Also supports the official template's point= format:
    Action: click(point='<point>452 128</point>')
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from alchemy.schemas import ActionTier, VisionAction


class CoordMode(str, Enum):
    """How model coordinates map to screen pixels."""
    NORMALIZED = "normalized"  # 0-1000 relative (UI-TARS v1 / 72B-DPO)
    ABSOLUTE = "absolute"  # Pixel coords in resized image space (UI-TARS v1.5)


# --- Regex patterns ---
_THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?=\nAction:|\Z)", re.DOTALL)
# v1.5 reflection mode: Reflection: ... Action_Summary: ... Action: ...
_REFLECTION_RE = re.compile(r"Reflection:\s*(.+?)(?=\nAction_Summary:|\nAction:|\Z)", re.DOTALL)
_ACTION_SUMMARY_RE = re.compile(r"Action_Summary:\s*(.+?)(?=\nAction:|\Z)", re.DOTALL)
_ACTION_RE = re.compile(r"Action:\s*(\w+)\((.*)\)\s*$", re.MULTILINE)
_COORD_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
_CONTENT_RE = re.compile(r"content='((?:[^'\\]|\\.)*)'")
_KEY_RE = re.compile(r"key='([^']*)'")
_DIRECTION_RE = re.compile(r"direction='([^']*)'")
_AMOUNT_RE = re.compile(r"amount=(\d+)")
# Handles both formats:
#   start_box='(452,128)'                          — simple format
#   start_box='<|box_start|>(13,980)<|box_end|>'   — native format
_START_BOX_RE = re.compile(r"start_box='([^']+)'")
_END_BOX_RE = re.compile(r"end_box='([^']+)'")
# Official v1 point= format: point='<point>452 128</point>'
_POINT_RE = re.compile(r"point='([^']+)'")
_POINT_COORDS_RE = re.compile(r"<point>\s*(\d+)\s+(\d+)\s*</point>")
# start_point / end_point for drag in official format
_START_POINT_RE = re.compile(r"start_point='([^']+)'")
_END_POINT_RE = re.compile(r"end_point='([^']+)'")

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


def _extract_coords(raw_value: str) -> tuple[int, int] | None:
    """Extract (x, y) from any coordinate format.

    Handles:
        (452,128)
        <|box_start|>(13,980)<|box_end|>
        <point>452 128</point>
    """
    # Try <point>x y</point> first (official v1 format)
    point_match = _POINT_COORDS_RE.search(raw_value)
    if point_match:
        return (int(point_match.group(1)), int(point_match.group(2)))

    # Try (x, y) format (works for both simple and box_start/box_end wrapped)
    coord_match = _COORD_RE.search(raw_value)
    if coord_match:
        return (int(coord_match.group(1)), int(coord_match.group(2)))

    return None


def _extract_start_coords(args_str: str) -> tuple[int, int] | None:
    """Extract start coordinates from any argument format."""
    # Try start_box= first
    match = _START_BOX_RE.search(args_str)
    if match:
        return _extract_coords(match.group(1))

    # Try point= (official v1 template)
    match = _POINT_RE.search(args_str)
    if match:
        return _extract_coords(match.group(1))

    # Try start_point= (official v1 drag)
    match = _START_POINT_RE.search(args_str)
    if match:
        return _extract_coords(match.group(1))

    return None


def _extract_end_coords(args_str: str) -> tuple[int, int] | None:
    """Extract end coordinates from any argument format."""
    # Try end_box= first
    match = _END_BOX_RE.search(args_str)
    if match:
        return _extract_coords(match.group(1))

    # Try end_point= (official v1 drag)
    match = _END_POINT_RE.search(args_str)
    if match:
        return _extract_coords(match.group(1))

    return None


def parse_uitars_response(raw: str) -> ParsedAction:
    """Extract Thought and Action from UI-TARS raw output.

    Supports v1 (Thought:), v1.5 (Reflection: + Action_Summary:), and
    summary-only (Action_Summary:) formats.

    Raises:
        ValueError: If the response cannot be parsed.
    """
    # Try Reflection first (v1.5 mode), then Thought, then Action_Summary
    thought = ""
    reflection_match = _REFLECTION_RE.search(raw)
    if reflection_match:
        thought = reflection_match.group(1).strip()
        # Append action summary if present
        summary_match = _ACTION_SUMMARY_RE.search(raw)
        if summary_match:
            thought = f"{thought} | Plan: {summary_match.group(1).strip()}"
    else:
        thought_match = _THOUGHT_RE.search(raw)
        if thought_match:
            thought = thought_match.group(1).strip()
        else:
            summary_match = _ACTION_SUMMARY_RE.search(raw)
            if summary_match:
                thought = summary_match.group(1).strip()

    # Extract action
    action_match = _ACTION_RE.search(raw)
    if not action_match:
        raise ValueError(f"Could not parse action from: {raw!r}")

    action_type = action_match.group(1)
    args_str = action_match.group(2)

    # Parse coordinates
    start_box = _extract_start_coords(args_str)
    end_box = _extract_end_coords(args_str)

    # Parse content (for type, finished)
    content = None
    content_match = _CONTENT_RE.search(args_str)
    if content_match:
        # Handle escape sequences
        raw_content = content_match.group(1)
        content = raw_content.replace("\\'", "'").replace('\\"', '"')

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


def scale_coord_normalized(
    norm_x: int, norm_y: int, width: int, height: int,
) -> tuple[int, int]:
    """Convert normalized (0-1000) coordinates to pixel coordinates."""
    px = round(norm_x / 1000 * width)
    py = round(norm_y / 1000 * height)
    return (min(max(px, 0), width), min(max(py, 0), height))


def scale_coord_absolute(
    abs_x: int, abs_y: int,
    resized_width: int, resized_height: int,
    screen_width: int, screen_height: int,
) -> tuple[int, int]:
    """Convert absolute coordinates from resized image space to screen pixels.

    UI-TARS-1.5 outputs pixel coordinates relative to the smart-resized image.
    We map those back to the original screen resolution.
    """
    if resized_width == 0 or resized_height == 0:
        return (abs_x, abs_y)
    px = round(abs_x / resized_width * screen_width)
    py = round(abs_y / resized_height * screen_height)
    return (min(max(px, 0), screen_width), min(max(py, 0), screen_height))


# Backward-compatible alias
scale_coord = scale_coord_normalized


def smart_resize_dimensions(
    width: int, height: int,
    factor: int = 28,
    min_pixels: int = 100 * 28 * 28,
    max_pixels: int = 16384 * 28 * 28,
) -> tuple[int, int]:
    """Calculate the smart-resized dimensions that UI-TARS-1.5 uses internally.

    The model processes images at these dimensions, so absolute coordinates
    are relative to this size.
    """
    total = width * height
    if total < min_pixels:
        scale = (min_pixels / total) ** 0.5
    elif total > max_pixels:
        scale = (max_pixels / total) ** 0.5
    else:
        scale = 1.0

    new_w = int(width * scale)
    new_h = int(height * scale)

    # Round to nearest multiple of factor
    new_w = max(factor, (new_w + factor // 2) // factor * factor)
    new_h = max(factor, (new_h + factor // 2) // factor * factor)

    return (new_w, new_h)


def validate_coords(
    x: int | None, y: int | None, max_x: int, max_y: int,
) -> tuple[int | None, int | None]:
    """Clamp coordinates to valid screen bounds."""
    if x is not None:
        x = min(max(x, 0), max_x)
    if y is not None:
        y = min(max(y, 0), max_y)
    return x, y


def to_vision_action(
    parsed: ParsedAction,
    screen_width: int = 1920,
    screen_height: int = 1080,
    coord_mode: CoordMode = CoordMode.NORMALIZED,
    resized_width: int = 0,
    resized_height: int = 0,
) -> VisionAction:
    """Convert ParsedAction to VisionAction with pixel coordinates.

    Args:
        parsed: Parsed model output.
        screen_width: Actual screen resolution width.
        screen_height: Actual screen resolution height.
        coord_mode: How to interpret model coordinates.
        resized_width: Width of smart-resized image (for ABSOLUTE mode).
        resized_height: Height of smart-resized image (for ABSOLUTE mode).
    """
    action_name = _ACTION_MAP.get(parsed.action_type, parsed.action_type)

    x, y = None, None
    if parsed.start_box:
        if coord_mode == CoordMode.ABSOLUTE and resized_width > 0:
            x, y = scale_coord_absolute(
                *parsed.start_box, resized_width, resized_height,
                screen_width, screen_height,
            )
        else:
            x, y = scale_coord_normalized(*parsed.start_box, screen_width, screen_height)

    end_x, end_y = None, None
    if parsed.end_box:
        if coord_mode == CoordMode.ABSOLUTE and resized_width > 0:
            end_x, end_y = scale_coord_absolute(
                *parsed.end_box, resized_width, resized_height,
                screen_width, screen_height,
            )
        else:
            end_x, end_y = scale_coord_normalized(*parsed.end_box, screen_width, screen_height)

    # Validate bounds
    x, y = validate_coords(x, y, screen_width, screen_height)
    end_x, end_y = validate_coords(end_x, end_y, screen_width, screen_height)

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
    """Classify action tier based on action type (base rules)."""
    if action.action in ("wait", "done", "fail"):
        return ActionTier.AUTO
    if action.action in ("type", "hotkey"):
        return ActionTier.NOTIFY
    if action.action == "scroll":
        return ActionTier.AUTO
    return ActionTier.AUTO
