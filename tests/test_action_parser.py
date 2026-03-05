"""Action parser tests — parse UI-TARS output into VisionAction."""

import pytest

from alchemy.click.action_parser import (
    ParsedAction,
    classify_tier,
    parse_uitars_response,
    scale_coord,
    to_vision_action,
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
        action = to_vision_action(parsed, 1920, 1080)
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
