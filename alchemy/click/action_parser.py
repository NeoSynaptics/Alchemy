"""Shim — action parser moved to alchemy.click.flow.action_parser."""
from alchemy.click.flow.action_parser import (  # noqa: F401
    CoordMode,
    ParsedAction,
    classify_tier,
    parse_uitars_response,
    scale_coord,
    scale_coord_absolute,
    scale_coord_normalized,
    smart_resize_dimensions,
    to_vision_action,
    validate_coords,
)
