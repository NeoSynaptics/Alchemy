"""Action parser tests — parse UI-TARS output into VisionAction."""

import pytest

from alchemy.click.action_parser import (
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
from alchemy.schemas import ActionTier


class TestParseUitarsResponse:
    def test_click(self):
        raw = "Thought: I see a search box.\nAction: click(start_box='(452,128)')"
        p = parse_uitars_response(raw)
        assert p.action_type == "click"
        assert p.start_box == (452, 128)
        assert "search box" in p.thought

    def test_double_click(self):
        raw = "Thought: Open the file.\nAction: left_double(start_box='(300,400)')"
        p = parse_uitars_response(raw)
        assert p.action_type == "left_double"
        assert p.start_box == (300, 400)

    def test_right_click(self):
        raw = "Thought: Open context menu.\nAction: right_single(start_box='(500,500)')"
        p = parse_uitars_response(raw)
        assert p.action_type == "right_single"

    def test_drag(self):
        raw = "Thought: Drag file.\nAction: drag(start_box='(100,200)', end_box='(300,400)')"
        p = parse_uitars_response(raw)
        assert p.action_type == "drag"
        assert p.start_box == (100, 200)
        assert p.end_box == (300, 400)

    def test_type(self):
        raw = "Thought: Type search query.\nAction: type(content='hello world')"
        p = parse_uitars_response(raw)
        assert p.action_type == "type"
        assert p.content == "hello world"

    def test_hotkey(self):
        raw = "Thought: Copy text.\nAction: hotkey(key='ctrl+c')"
        p = parse_uitars_response(raw)
        assert p.action_type == "hotkey"
        assert p.key == "ctrl+c"

    def test_scroll(self):
        raw = "Thought: Scroll down.\nAction: scroll(start_box='(500,500)', direction='down', amount=3)"
        p = parse_uitars_response(raw)
        assert p.action_type == "scroll"
        assert p.start_box == (500, 500)
        assert p.direction == "down"
        assert p.amount == 3

    def test_wait(self):
        raw = "Thought: Page loading.\nAction: wait()"
        p = parse_uitars_response(raw)
        assert p.action_type == "wait"

    def test_finished(self):
        raw = "Thought: Task done.\nAction: finished(content='Search complete')"
        p = parse_uitars_response(raw)
        assert p.action_type == "finished"
        assert p.content == "Search complete"

    def test_no_thought(self):
        raw = "Action: click(start_box='(100,200)')"
        p = parse_uitars_response(raw)
        assert p.thought == ""
        assert p.action_type == "click"

    def test_multiline_thought(self):
        raw = "Thought: I see a complex form\nwith multiple fields.\nAction: click(start_box='(100,200)')"
        p = parse_uitars_response(raw)
        assert "complex form" in p.thought
        assert p.action_type == "click"

    def test_malformed_raises(self):
        with pytest.raises(ValueError):
            parse_uitars_response("Just some random text")

    def test_extra_whitespace(self):
        raw = "  Thought:  I see it.  \n  Action:  click(start_box='( 452 , 128 )')  "
        p = parse_uitars_response(raw)
        assert p.action_type == "click"
        assert p.start_box == (452, 128)

    def test_72b_box_tokens(self):
        """72B-DPO uses <|box_start|>...<|box_end|> coordinate tokens."""
        raw = "Thought: Click the icon.\nAction: click(start_box='<|box_start|>(13,980)<|box_end|>')"
        p = parse_uitars_response(raw)
        assert p.action_type == "click"
        assert p.start_box == (13, 980)

    def test_72b_drag_box_tokens(self):
        raw = "Thought: Drag it.\nAction: drag(start_box='<|box_start|>(100,200)<|box_end|>', end_box='<|box_start|>(800,900)<|box_end|>')"
        p = parse_uitars_response(raw)
        assert p.action_type == "drag"
        assert p.start_box == (100, 200)
        assert p.end_box == (800, 900)


class TestScaleCoord:
    def test_center(self):
        assert scale_coord(500, 500, 1920, 1080) == (960, 540)

    def test_origin(self):
        assert scale_coord(0, 0, 1920, 1080) == (0, 0)

    def test_max(self):
        assert scale_coord(1000, 1000, 1920, 1080) == (1920, 1080)

    def test_arbitrary(self):
        x, y = scale_coord(452, 128, 1920, 1080)
        assert x == round(452 / 1000 * 1920)
        assert y == round(128 / 1000 * 1080)


class TestToVisionAction:
    def test_click_scaled(self):
        parsed = ParsedAction(thought="Click it", action_type="click", start_box=(500, 500))
        action = to_vision_action(parsed, 1920, 1080, coord_mode=CoordMode.NORMALIZED)
        assert action.action == "click"
        assert action.x == 960
        assert action.y == 540
        assert action.reasoning == "Click it"

    def test_drag(self):
        parsed = ParsedAction(
            thought="Drag", action_type="drag",
            start_box=(100, 200), end_box=(800, 900),
        )
        action = to_vision_action(parsed, 1920, 1080)
        assert action.action == "drag"
        assert action.end_x is not None
        assert action.end_y is not None

    def test_type_maps_content(self):
        parsed = ParsedAction(thought="", action_type="type", content="hello")
        action = to_vision_action(parsed)
        assert action.action == "type"
        assert action.text == "hello"

    def test_finished_maps_to_done(self):
        parsed = ParsedAction(thought="Done", action_type="finished", content="Complete")
        action = to_vision_action(parsed)
        assert action.action == "done"
        assert action.text == "Complete"

    def test_scroll_fields(self):
        parsed = ParsedAction(
            thought="", action_type="scroll",
            start_box=(500, 500), direction="down", amount=5,
        )
        action = to_vision_action(parsed)
        assert action.direction == "down"
        assert action.amount == 5


class TestClassifyTier:
    def test_click_auto(self):
        from alchemy.schemas import VisionAction
        action = VisionAction(action="click", x=100, y=200)
        assert classify_tier(action) == ActionTier.AUTO

    def test_type_notify(self):
        from alchemy.schemas import VisionAction
        action = VisionAction(action="type", text="hello")
        assert classify_tier(action) == ActionTier.NOTIFY

    def test_wait_auto(self):
        from alchemy.schemas import VisionAction
        action = VisionAction(action="wait")
        assert classify_tier(action) == ActionTier.AUTO

    def test_done_auto(self):
        from alchemy.schemas import VisionAction
        action = VisionAction(action="done")
        assert classify_tier(action) == ActionTier.AUTO


# --- v2 features: point format, reflection mode, coordinate modes, validation ---


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
        x, y = scale_coord_absolute(960, 540, 1920, 1080, 1920, 1080)
        assert (x, y) == (960, 540)

    def test_upscale(self):
        x, y = scale_coord_absolute(640, 360, 1280, 720, 1920, 1080)
        assert (x, y) == (960, 540)

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
        assert w * h <= 16384 * 28 * 28

    def test_720p(self):
        w, h = smart_resize_dimensions(1280, 720)
        assert w % 28 == 0
        assert h % 28 == 0

    def test_very_small(self):
        w, h = smart_resize_dimensions(100, 100)
        assert w >= 28
        assert h >= 28
        assert w * h >= 100 * 28 * 28


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
        assert action.x == round(100 / 1000 * 1920)

    def test_bounds_validated(self):
        parsed = ParsedAction(thought="Click", action_type="click", start_box=(1500, 1500))
        action = to_vision_action(parsed, 1920, 1080)
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
