"""Parse vision model output into structured VisionAction objects.

Primary format (PROVEN — Qwen2.5-VL native):
    Thought: I see a search box.
    Action: click {"point_2d": [588, 44]}

Coordinates are IMAGE PIXEL values (in the 1280x720 capture), scaled to screen.
Scaling: screen_x = point_x / image_width * screen_width

Also supports legacy formats for fallback:
    Action: click(start_box='(452,128)')  — start_box with (x,y)
    Action: click@(436,15)                — at-sign format
    Action: type "hello"                  — text actions
    Action: hotkey ctrl+c                 — keyboard shortcuts
    Action: scroll up/down                — scrolling
    Action: done                          — terminal
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum

from alchemy.schemas import ActionTier, VisionAction


class CoordMode(str, Enum):
    """How model coordinates map to screen pixels."""
    IMAGE_PIXEL = "image_pixel"  # Pixel coords in captured image (Qwen2.5-VL native)
    NORMALIZED = "normalized"  # 0-1000 relative (UI-TARS v1 legacy)
    ABSOLUTE = "absolute"  # Pixel coords in resized image space (UI-TARS v1.5 legacy)


# --- Regex patterns ---
_THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?=\nAction:|\Z)", re.DOTALL)
_REFLECTION_RE = re.compile(r"Reflection:\s*(.+?)(?=\nAction_Summary:|\nAction:|\Z)", re.DOTALL)
_ACTION_SUMMARY_RE = re.compile(r"Action_Summary:\s*(.+?)(?=\nAction:|\Z)", re.DOTALL)
_ACTION_RE = re.compile(r"Action:\s*(\w+)\((.*)\)\s*$", re.MULTILINE)
_COORD_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
_CONTENT_RE = re.compile(r"content='((?:[^'\\]|\\.)*)'")
_KEY_RE = re.compile(r"key='([^']*)'")
_DIRECTION_RE = re.compile(r"direction='([^']*)'")
_AMOUNT_RE = re.compile(r"amount=(\d+)")
_START_BOX_RE = re.compile(r"start_box='([^']+)'")
_END_BOX_RE = re.compile(r"end_box='([^']+)'")
_POINT_RE = re.compile(r"point='([^']+)'")
_POINT_COORDS_RE = re.compile(r"<point>\s*(\d+)\s+(\d+)\s*</point>")
_START_POINT_RE = re.compile(r"start_point='([^']+)'")
_END_POINT_RE = re.compile(r"end_point='([^']+)'")

# Qwen2.5-VL native JSON: {"point_2d": [588, 44]}
_POINT_JSON_RE = re.compile(r'\{[^{}]*"point_2d"\s*:\s*\[([^\]]+)\][^{}]*\}')
# Action name from "Action: click" (before JSON or parens)
_ACTION_NAME_RE = re.compile(r"Action:\s*(\w+)")
# Action: type "text" or Action: type 'text'
_ACTION_TYPE_TEXT_RE = re.compile(r'Action:\s*type\s+["\'](.+?)["\']', re.MULTILINE)
# Action: hotkey key1+key2
_ACTION_HOTKEY_RE = re.compile(r"Action:\s*hotkey\s+(.+?)$", re.MULTILINE)
# Action: scroll up/down
_ACTION_SCROLL_RE = re.compile(r"Action:\s*scroll\s+(up|down|left|right)", re.MULTILINE)
# Action: click@(x,y) — minicpm-v format
_ACTION_AT_RE = re.compile(r"Action:\s*(\w+)@\((\d+)\s*,\s*(\d+)", re.MULTILINE)

# Map action names to VisionAction.action values
_ACTION_MAP = {
    "click": "click",
    "left_double": "double_click",
    "double_click": "double_click",
    "right_single": "right_click",
    "right_click": "right_click",
    "drag": "drag",
    "type": "type",
    "hotkey": "hotkey",
    "scroll": "scroll",
    "wait": "wait",
    "finished": "done",
    "done": "done",
}


@dataclass
class ParsedAction:
    """Intermediate parsed representation from model output."""
    thought: str
    action_type: str
    start_box: tuple[int, int] | None = None
    end_box: tuple[int, int] | None = None
    content: str | None = None
    key: str | None = None
    direction: str | None = None
    amount: int | None = None


# --- Coordinate extraction helpers ---

def _extract_coords(raw_value: str) -> tuple[int, int] | None:
    """Extract (x, y) from any coordinate format."""
    point_match = _POINT_COORDS_RE.search(raw_value)
    if point_match:
        return (int(point_match.group(1)), int(point_match.group(2)))
    coord_match = _COORD_RE.search(raw_value)
    if coord_match:
        return (int(coord_match.group(1)), int(coord_match.group(2)))
    return None


def _extract_start_coords(args_str: str) -> tuple[int, int] | None:
    for regex in (_START_BOX_RE, _POINT_RE, _START_POINT_RE):
        match = regex.search(args_str)
        if match:
            return _extract_coords(match.group(1))
    return None


def _extract_end_coords(args_str: str) -> tuple[int, int] | None:
    for regex in (_END_BOX_RE, _END_POINT_RE):
        match = regex.search(args_str)
        if match:
            return _extract_coords(match.group(1))
    return None


def _extract_thought(raw: str) -> str:
    """Extract thought from Reflection, Thought, or Action_Summary."""
    reflection_match = _REFLECTION_RE.search(raw)
    if reflection_match:
        thought = reflection_match.group(1).strip()
        summary_match = _ACTION_SUMMARY_RE.search(raw)
        if summary_match:
            thought = f"{thought} | Plan: {summary_match.group(1).strip()}"
        return thought
    thought_match = _THOUGHT_RE.search(raw)
    if thought_match:
        return thought_match.group(1).strip()
    summary_match = _ACTION_SUMMARY_RE.search(raw)
    if summary_match:
        return summary_match.group(1).strip()
    return ""


# --- Primary parser: Qwen2.5-VL point_2d JSON ---

def _parse_point_2d(raw: str, thought: str) -> ParsedAction | None:
    """Try Qwen2.5-VL native point_2d JSON format (most accurate).

    Handles: Action: click {"point_2d": [588, 44]}
    """
    action_name_match = _ACTION_NAME_RE.search(raw)
    action_name = action_name_match.group(1) if action_name_match else None

    # Terminal actions
    if action_name in ("done", "finished"):
        return ParsedAction(thought=thought, action_type="finished")
    if action_name == "wait":
        return ParsedAction(thought=thought, action_type="wait")

    # Text actions
    type_match = _ACTION_TYPE_TEXT_RE.search(raw)
    if type_match:
        return ParsedAction(thought=thought, action_type="type", content=type_match.group(1))

    hotkey_match = _ACTION_HOTKEY_RE.search(raw)
    if hotkey_match:
        return ParsedAction(thought=thought, action_type="hotkey", key=hotkey_match.group(1).strip())

    scroll_match = _ACTION_SCROLL_RE.search(raw)
    if scroll_match:
        return ParsedAction(thought=thought, action_type="scroll", direction=scroll_match.group(1))

    # point_2d JSON
    json_match = _POINT_JSON_RE.search(raw)
    if not json_match:
        return None

    try:
        data = json.loads(json_match.group(0))
        coords = data.get("point_2d", [])
        if len(coords) < 2:
            return None
        x = int(float(coords[0]))
        y = int(float(coords[1]))
        act = _ACTION_MAP.get(action_name, "click") if action_name else "click"
        return ParsedAction(thought=thought, action_type=act, start_box=(x, y))
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


# --- Main parse function ---

def parse_response(raw: str) -> ParsedAction:
    """Parse vision model response — tries point_2d first, then legacy formats.

    Priority:
    1. Qwen2.5-VL native point_2d JSON (most accurate)
    2. click@(x,y) format
    3. UI-TARS start_box/point format

    Raises:
        ValueError: If the response cannot be parsed.
    """
    thought = _extract_thought(raw)

    # 1. Try point_2d JSON first (Qwen2.5-VL native)
    parsed = _parse_point_2d(raw, thought)
    if parsed is not None:
        return parsed

    # 2. Try click@(x,y) format
    at_match = _ACTION_AT_RE.search(raw)
    if at_match:
        action_type = _ACTION_MAP.get(at_match.group(1), at_match.group(1))
        x, y = int(at_match.group(2)), int(at_match.group(3))
        return ParsedAction(thought=thought, action_type=action_type, start_box=(x, y))

    # 3. Try UI-TARS start_box format (legacy)
    action_match = _ACTION_RE.search(raw)
    if not action_match:
        raise ValueError(f"Could not parse action from: {raw!r}")

    action_type = action_match.group(1)
    args_str = action_match.group(2)

    start_box = _extract_start_coords(args_str)
    end_box = _extract_end_coords(args_str)

    content = None
    content_match = _CONTENT_RE.search(args_str)
    if content_match:
        content = content_match.group(1).replace("\\'", "'").replace('\\"', '"')

    key = None
    key_match = _KEY_RE.search(args_str)
    if key_match:
        key = key_match.group(1)

    direction = None
    dir_match = _DIRECTION_RE.search(args_str)
    if dir_match:
        direction = dir_match.group(1)

    amount = None
    amount_match = _AMOUNT_RE.search(args_str)
    if amount_match:
        amount = int(amount_match.group(1))

    return ParsedAction(
        thought=thought, action_type=action_type,
        start_box=start_box, end_box=end_box,
        content=content, key=key, direction=direction, amount=amount,
    )


# Backward-compatible alias
parse_uitars_response = parse_response


# --- Coordinate scaling ---

def scale_coord_image_pixel(
    img_x: int, img_y: int,
    image_width: int, image_height: int,
    screen_width: int, screen_height: int,
) -> tuple[int, int]:
    """Convert image pixel coordinates to screen pixels.

    Qwen2.5-VL outputs coords in the captured image space (e.g. 1280x720).
    Scale to actual screen resolution (e.g. 1920x1080).
    """
    if image_width == 0 or image_height == 0:
        return (img_x, img_y)
    px = round(img_x / image_width * screen_width)
    py = round(img_y / image_height * screen_height)
    return (min(max(px, 0), screen_width), min(max(py, 0), screen_height))


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
    """Convert absolute coordinates from resized image space to screen pixels."""
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
    """Calculate the smart-resized dimensions that UI-TARS-1.5 uses internally."""
    total = width * height
    if total < min_pixels:
        scale = (min_pixels / total) ** 0.5
    elif total > max_pixels:
        scale = (max_pixels / total) ** 0.5
    else:
        scale = 1.0
    new_w = int(width * scale)
    new_h = int(height * scale)
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


# --- Action conversion ---

def to_vision_action(
    parsed: ParsedAction,
    screen_width: int = 1920,
    screen_height: int = 1080,
    coord_mode: CoordMode = CoordMode.IMAGE_PIXEL,
    image_width: int = 1280,
    image_height: int = 720,
    resized_width: int = 0,
    resized_height: int = 0,
) -> VisionAction:
    """Convert ParsedAction to VisionAction with screen pixel coordinates."""
    action_name = _ACTION_MAP.get(parsed.action_type, parsed.action_type)

    x, y = None, None
    if parsed.start_box:
        if coord_mode == CoordMode.IMAGE_PIXEL:
            x, y = scale_coord_image_pixel(
                *parsed.start_box, image_width, image_height,
                screen_width, screen_height,
            )
        elif coord_mode == CoordMode.ABSOLUTE and resized_width > 0:
            x, y = scale_coord_absolute(
                *parsed.start_box, resized_width, resized_height,
                screen_width, screen_height,
            )
        else:
            x, y = scale_coord_normalized(*parsed.start_box, screen_width, screen_height)

    end_x, end_y = None, None
    if parsed.end_box:
        if coord_mode == CoordMode.IMAGE_PIXEL:
            end_x, end_y = scale_coord_image_pixel(
                *parsed.end_box, image_width, image_height,
                screen_width, screen_height,
            )
        elif coord_mode == CoordMode.ABSOLUTE and resized_width > 0:
            end_x, end_y = scale_coord_absolute(
                *parsed.end_box, resized_width, resized_height,
                screen_width, screen_height,
            )
        else:
            end_x, end_y = scale_coord_normalized(*parsed.end_box, screen_width, screen_height)

    x, y = validate_coords(x, y, screen_width, screen_height)
    end_x, end_y = validate_coords(end_x, end_y, screen_width, screen_height)

    text = parsed.content or parsed.key

    return VisionAction(
        action=action_name,
        x=x, y=y, end_x=end_x, end_y=end_y,
        text=text, reasoning=parsed.thought,
        tier=ActionTier.AUTO,
        direction=parsed.direction, amount=parsed.amount,
    )


def classify_tier(action: VisionAction) -> ActionTier:
    """Classify action tier based on action type (base rules)."""
    if action.action in ("wait", "done", "fail"):
        return ActionTier.AUTO
    if action.action in ("type", "hotkey"):
        return ActionTier.NOTIFY
    return ActionTier.AUTO
