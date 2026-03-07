"""AlchemyFlow — Vision/desktop path for native GUI automation.

Screenshot → Qwen2.5-VL 7B → pixel coordinates → ghost cursor → SendInput.
Works on any visible UI element on any Windows app.

The orange ghost cursor glides smoothly to each target, flashes on click,
and parks in the bottom-right corner when idle.
"""

from alchemy.click.flow.flow_agent import FlowAgent, ScreenSource, ActionSink, StepResult
from alchemy.click.flow.action_executor import ActionExecutor
from alchemy.click.flow.action_parser import (
    CoordMode,
    ParsedAction,
    classify_tier,
    parse_response,
    parse_uitars_response,
    scale_coord,
    scale_coord_absolute,
    scale_coord_image_pixel,
    scale_coord_normalized,
    smart_resize_dimensions,
    to_vision_action,
    validate_coords,
)
from alchemy.click.flow.omniparser import OmniParser, ParseResult, UIElement
from alchemy.click.flow.vision_agent import VisionAgent

__all__ = [
    "FlowAgent",
    "OmniParser",
    "ParseResult",
    "UIElement",
    "ScreenSource",
    "ActionSink",
    "StepResult",
    "ActionExecutor",
    "CoordMode",
    "ParsedAction",
    "VisionAgent",
    "classify_tier",
    "parse_response",
    "parse_uitars_response",
    "scale_coord",
    "scale_coord_absolute",
    "scale_coord_image_pixel",
    "scale_coord_normalized",
    "smart_resize_dimensions",
    "to_vision_action",
    "validate_coords",
]
