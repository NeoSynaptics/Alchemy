"""Tests for Qwen2.5-VL native JSON point parsing — escalation + desktop agent.

Qwen2.5-VL outputs pixel coordinates in JSON format:
    {"point_2d": [452, 128], "label": "search button"}

These tests verify that both the escalation module and desktop agent
can parse this format as a fallback when the standard Thought/Action format
is not used by the model.
"""

import pytest

from alchemy.core.escalation import (
    EscalationResult,
    _parse_escalation_response,
    _parse_qwen_vl_json,
)
from alchemy.desktop.agent import (
    _parse_response as desktop_parse_response,
    _parse_qwen_vl_json as desktop_parse_qwen_json,
)


# --- Escalation: Qwen2.5-VL JSON parsing ---


class TestEscalationQwenVLJSON:
    def test_native_json_point(self):
        """Qwen2.5-VL native format parses as click action."""
        raw = '{"point_2d": [452, 128], "label": "search button"}'
        result = _parse_escalation_response(raw, 1280, 720)
        assert result.success is True
        assert result.action_type == "click"
        assert result.x == 452
        assert result.y == 128
        assert "search button" in result.thought

    def test_json_point_with_surrounding_text(self):
        """JSON embedded in natural language text."""
        raw = 'I can see the search button. {"point_2d": [640, 360], "label": "search"}'
        result = _parse_escalation_response(raw, 1280, 720)
        assert result.success is True
        assert result.x == 640
        assert result.y == 360

    def test_json_point_with_thought(self):
        """Thought line + JSON point (no Action: line)."""
        raw = 'Thought: I see the search button at the top.\n{"point_2d": [300, 50], "label": "search"}'
        result = _parse_escalation_response(raw, 1280, 720)
        assert result.success is True
        assert result.x == 300
        assert result.y == 50
        assert "search button" in result.thought

    def test_json_point_clamped(self):
        """Out-of-bounds coordinates get clamped."""
        raw = '{"point_2d": [1500, 800], "label": "edge"}'
        result = _parse_escalation_response(raw, 1280, 720)
        assert result.x == 1280
        assert result.y == 720

    def test_json_point_float_coords(self):
        """Qwen2.5-VL sometimes outputs float coords."""
        raw = '{"point_2d": [452.7, 128.3], "label": "button"}'
        result = _parse_escalation_response(raw, 1280, 720)
        assert result.x == 452  # int() truncates
        assert result.y == 128

    def test_standard_format_takes_priority(self):
        """Standard Action: format is tried before JSON fallback."""
        raw = (
            'Thought: Click search.\n'
            'Action: click(start_box="(100,200)")\n'
            '{"point_2d": [999, 999], "label": "ignored"}'
        )
        result = _parse_escalation_response(raw, 1280, 720)
        assert result.x == 100  # Standard format wins
        assert result.y == 200

    def test_no_parseable_format_raises(self):
        """Neither Action: nor JSON point raises ValueError."""
        with pytest.raises(ValueError, match="No Action: or JSON point"):
            _parse_escalation_response("Just random text without actions", 1280, 720)

    def test_empty_point_array_raises(self):
        """Empty point_2d array should not parse."""
        raw = '{"point_2d": [], "label": "nothing"}'
        with pytest.raises(ValueError):
            _parse_escalation_response(raw, 1280, 720)

    def test_invalid_json_raises(self):
        """Malformed JSON should not parse."""
        raw = '{"point_2d": [broken'
        with pytest.raises(ValueError):
            _parse_escalation_response(raw, 1280, 720)


class TestEscalationParseQwenJsonDirect:
    """Direct tests for _parse_qwen_vl_json helper."""

    def test_valid_json(self):
        result = _parse_qwen_vl_json(
            '{"point_2d": [500, 300], "label": "ok button"}',
            "click it", 1280, 720,
        )
        assert result is not None
        assert result.action_type == "click"
        assert result.x == 500
        assert result.y == 300

    def test_no_json(self):
        result = _parse_qwen_vl_json("no json here", "", 1280, 720)
        assert result is None

    def test_json_without_point_2d(self):
        result = _parse_qwen_vl_json('{"bbox_2d": [0, 0, 100, 100]}', "", 1280, 720)
        assert result is None


# --- Desktop Agent: Qwen2.5-VL JSON parsing ---


class TestDesktopQwenVLJSON:
    def test_native_json_point(self):
        """Desktop parser handles Qwen2.5-VL JSON format."""
        raw = '{"point_2d": [640, 360], "label": "start menu"}'
        action, x, y, text, direction, thought = desktop_parse_response(raw)
        assert action == "click"
        assert x == 640
        assert y == 360

    def test_json_with_thought(self):
        raw = 'Thought: I see the start menu.\n{"point_2d": [50, 700], "label": "start"}'
        action, x, y, text, direction, thought = desktop_parse_response(raw)
        assert action == "click"
        assert x == 50
        assert y == 700
        assert "start menu" in thought

    def test_standard_action_takes_priority(self):
        """Standard Action: format wins over JSON."""
        raw = (
            'Thought: Click start.\n'
            'Action: click(start_box="(100,200)")\n'
            '{"point_2d": [999, 999]}'
        )
        action, x, y, text, direction, thought = desktop_parse_response(raw)
        assert x == 100
        assert y == 200

    def test_click_at_format_takes_priority(self):
        """click@(X,Y) format wins over JSON."""
        raw = 'Thought: Click.\nAction: click@(300,400)\n{"point_2d": [999, 999]}'
        action, x, y, text, direction, thought = desktop_parse_response(raw)
        assert x == 300
        assert y == 400

    def test_no_parseable_format_raises(self):
        with pytest.raises(ValueError, match="No Action:"):
            desktop_parse_response("Completely unparseable text")

    def test_json_float_coords(self):
        raw = '{"point_2d": [320.5, 240.8], "label": "icon"}'
        action, x, y, text, direction, thought = desktop_parse_response(raw)
        assert action == "click"
        assert x == 320  # int() truncates
        assert y == 240


class TestDesktopParseQwenJsonDirect:
    """Direct tests for desktop _parse_qwen_vl_json helper."""

    def test_valid_json(self):
        result = desktop_parse_qwen_json(
            '{"point_2d": [500, 300], "label": "ok"}', "thought",
        )
        assert result is not None
        assert result == ("click", 500, 300, None, None, "thought")

    def test_no_json(self):
        result = desktop_parse_qwen_json("no json", "")
        assert result is None

    def test_uses_label_as_thought_fallback(self):
        result = desktop_parse_qwen_json(
            '{"point_2d": [100, 200], "label": "menu icon"}', "",
        )
        assert result is not None
        assert "menu icon" in result[5]  # thought field


# --- Existing format tests still pass (regression) ---


class TestExistingFormatsUnchanged:
    """Verify that existing Action: and click@() formats still work."""

    def test_escalation_standard_click(self):
        raw = 'Thought: Click.\nAction: click(start_box="(500,300)")'
        result = _parse_escalation_response(raw, 1280, 720)
        assert result.action_type == "click"
        assert result.x == 500
        assert result.y == 300

    def test_escalation_type_action(self):
        raw = 'Thought: Type.\nAction: type(content="hello")'
        result = _parse_escalation_response(raw, 1280, 720)
        assert result.action_type == "type"
        assert result.text == "hello"

    def test_escalation_finished(self):
        raw = 'Thought: Done.\nAction: finished(content="done")'
        result = _parse_escalation_response(raw, 1280, 720)
        assert result.action_type == "done"

    def test_desktop_standard_click(self):
        raw = 'Thought: Button.\nAction: click(start_box="(450,200)")'
        action, x, y, text, direction, thought = desktop_parse_response(raw)
        assert action == "click"
        assert x == 450

    def test_desktop_click_at_format(self):
        raw = "Thought: Click.\nAction: click@(463,158)"
        action, x, y, text, direction, thought = desktop_parse_response(raw)
        assert action == "click"
        assert x == 463

    def test_desktop_type_action(self):
        raw = 'Thought: Type.\nAction: type(content="hello")'
        action, x, y, text, direction, thought = desktop_parse_response(raw)
        assert action == "type"
        assert text == "hello"
