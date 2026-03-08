"""Tests for new action parser features — v1.5 support, coordinate modes, validation."""

import pytest

from alchemy.click.action_parser import (
    CoordMode,
    ParsedAction,
    parse_uitars_response,
    scale_coord_absolute,
    scale_coord_normalized,
    smart_resize_dimensions,
    to_vision_action,
    validate_coords,
)
from alchemy.schemas import ActionTier


class TestOfficialPointFormat:
    """Test the official UI-TARS v1 point= format from ByteDance's template."""

    def test_click_point(self):
        raw = "Thought: Click the search box.\nAction: click(point='<point>452 128</point>')"
        p = parse_uitars_response(raw)
        assert p.action_type == "click"
        assert p.start_box == (452, 128)

    def test_scroll_point(self):
        raw = "Thought: Scroll down.\nAction: scroll(point='<point>500 500</point>', direction='down')"
        p = parse_uitars_response(raw)
        assert p.action_type == "scroll"
        assert p.start_box == (500, 500)
        assert p.direction == "down"

    def test_drag_start_end_point(self):
        raw = "Thought: Drag file.\nAction: drag(start_point='<point>100 200</point>', end_point='<point>800 900</point>')"
        p = parse_uitars_response(raw)
        assert p.action_type == "drag"
        assert p.start_box == (100, 200)
        assert p.end_box == (800, 900)

    def test_point_with_extra_whitespace(self):
        raw = "Thought: Click.\nAction: click(point='<point>  452   128  </point>')"
        p = parse_uitars_response(raw)
        assert p.start_box == (452, 128)


class TestReflectionMode:
    """Test UI-TARS-1.5 Reflection: + Action_Summary: output format."""

    def test_reflection_with_summary(self):
        raw = (
            "Reflection: The previous click opened the menu successfully.\n"
            "Action_Summary: Now I need to click the search option.\n"
            "Action: click(start_box='(300,400)')"
        )
        p = parse_uitars_response(raw)
        assert "previous click opened" in p.thought
        assert "Plan:" in p.thought
        assert "search option" in p.thought
        assert p.action_type == "click"
        assert p.start_box == (300, 400)

    def test_action_summary_only(self):
        raw = (
            "Action_Summary: Click the play button.\n"
            "Action: click(start_box='(600,700)')"
        )
        p = parse_uitars_response(raw)
        assert "play button" in p.thought
        assert p.action_type == "click"

    def test_reflection_without_summary(self):
        raw = (
            "Reflection: The page loaded correctly.\n"
            "Action: click(start_box='(200,300)')"
        )
        p = parse_uitars_response(raw)
        assert "page loaded" in p.thought
        assert p.action_type == "click"


class TestContentEscapes:
    """Test escape character handling in type content."""

    def test_escaped_single_quote(self):
        raw = "Thought: Type it.\nAction: type(content='it\\'s working')"
        p = parse_uitars_response(raw)
        assert p.content == "it's working"

    def test_escaped_double_quote(self):
        raw = 'Thought: Type.\nAction: type(content=\'say \\\"hello\\\"\')'
        p = parse_uitars_response(raw)
        assert p.content == 'say "hello"'


class TestScaleCoordAbsolute:
    """Test absolute coordinate mapping for UI-TARS-1.5."""

    def test_identity(self):
        # If resized == screen, coords pass through
        x, y = scale_coord_absolute(960, 540, 1920, 1080, 1920, 1080)
        assert (x, y) == (960, 540)

    def test_upscale(self):
        # Resized to 1280x720 (downscaled), model outputs coords in that space
        x, y = scale_coord_absolute(640, 360, 1280, 720, 1920, 1080)
        assert (x, y) == (960, 540)  # Should map to center of 1920x1080

    def test_zero_resized_passthrough(self):
        x, y = scale_coord_absolute(100, 200, 0, 0, 1920, 1080)
        assert (x, y) == (100, 200)

    def test_clamps_to_screen(self):
        x, y = scale_coord_absolute(2000, 1500, 1280, 720, 1920, 1080)
        assert x <= 1920
        assert y <= 1080


class TestSmartResizeDimensions:
    def test_1080p(self):
        w, h = smart_resize_dimensions(1920, 1080)
        assert w % 28 == 0
        assert h % 28 == 0
        assert w * h <= 16384 * 28 * 28  # within max

    def test_720p(self):
        w, h = smart_resize_dimensions(1280, 720)
        assert w % 28 == 0
        assert h % 28 == 0

    def test_very_small(self):
        w, h = smart_resize_dimensions(100, 100)
        assert w >= 28
        assert h >= 28
        assert w * h >= 100 * 28 * 28  # scaled up to min


class TestValidateCoords:
    def test_clamp_negative(self):
        x, y = validate_coords(-10, -20, 1920, 1080)
        assert x == 0
        assert y == 0

    def test_clamp_over(self):
        x, y = validate_coords(2000, 1200, 1920, 1080)
        assert x == 1920
        assert y == 1080

    def test_none_passthrough(self):
        x, y = validate_coords(None, None, 1920, 1080)
        assert x is None
        assert y is None

    def test_valid_unchanged(self):
        x, y = validate_coords(500, 300, 1920, 1080)
        assert (x, y) == (500, 300)


class TestToVisionActionCoordModes:
    """Test to_vision_action with different coordinate modes."""

    def test_normalized_default(self):
        parsed = ParsedAction(thought="Click", action_type="click", start_box=(500, 500))
        action = to_vision_action(parsed, 1920, 1080, coord_mode=CoordMode.NORMALIZED)
        assert action.x == 960
        assert action.y == 540

    def test_absolute_mode(self):
        parsed = ParsedAction(thought="Click", action_type="click", start_box=(640, 360))
        action = to_vision_action(
            parsed, 1920, 1080,
            coord_mode=CoordMode.ABSOLUTE,
            resized_width=1280, resized_height=720,
        )
        assert action.x == 960
        assert action.y == 540

    def test_absolute_mode_without_resize_falls_back(self):
        parsed = ParsedAction(thought="Click", action_type="click", start_box=(100, 200))
        action = to_vision_action(
            parsed, 1920, 1080,
            coord_mode=CoordMode.ABSOLUTE,
            resized_width=0, resized_height=0,
        )
        # Without resize dimensions, falls back to normalized
        assert action.x == round(100 / 1000 * 1920)

    def test_bounds_validated(self):
        parsed = ParsedAction(thought="Click", action_type="click", start_box=(1500, 1500))
        action = to_vision_action(parsed, 1920, 1080)
        # 1500/1000 * 1920 = 2880 -> clamped to 1920
        assert action.x == 1920
        assert action.y == 1080


class TestScaleCoordNormalized:
    def test_clamps_negative(self):
        x, y = scale_coord_normalized(-10, -10, 1920, 1080)
        assert x == 0
        assert y == 0

    def test_clamps_over_1000(self):
        x, y = scale_coord_normalized(1100, 1100, 1920, 1080)
        assert x <= 1920
        assert y <= 1080
