"""Tests for Tier 1.5 vision escalation — stuck detection + UI-TARS 7B fallback."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from alchemy.playwright.escalation import (
    EscalationResult,
    StuckDetector,
    StuckReason,
    VisionEscalation,
    _parse_escalation_response,
)


# --- StuckDetector Tests ---


class TestStuckDetector:
    def test_no_issues_returns_none(self):
        d = StuckDetector()
        assert d.check() is None

    def test_parse_failures_trigger(self):
        d = StuckDetector(max_parse_failures=3)
        d.record_parse_failure()
        d.record_parse_failure()
        assert d.check() is None  # Only 2
        d.record_parse_failure()
        assert d.check() == StuckReason.PARSE_FAILURES

    def test_success_resets_parse_failures(self):
        d = StuckDetector(max_parse_failures=2)
        d.record_parse_failure()
        d.record_success("click@e1")
        d.record_parse_failure()
        assert d.check() is None  # Reset after success

    def test_action_loop_trigger(self):
        d = StuckDetector(max_repeated_actions=3)
        d.record_success("click@e5")
        d.record_success("click@e5")
        assert d.check() is None  # Only 2
        d.record_success("click@e5")
        assert d.check() == StuckReason.ACTION_LOOP

    def test_different_actions_no_loop(self):
        d = StuckDetector(max_repeated_actions=3)
        d.record_success("click@e1")
        d.record_success("click@e2")
        d.record_success("click@e3")
        assert d.check() is None

    def test_complexity_trigger(self):
        d = StuckDetector(complexity_threshold=60)
        assert d.check(ref_count=59) is None
        assert d.check(ref_count=61) == StuckReason.COMPLEXITY

    def test_reset_clears_all(self):
        d = StuckDetector(max_parse_failures=2, max_repeated_actions=2)
        d.record_parse_failure()
        d.record_parse_failure()
        d.record_success("click@e1")
        d.record_success("click@e1")
        d.reset()
        assert d.check() is None

    def test_loop_broken_by_different_action(self):
        d = StuckDetector(max_repeated_actions=3)
        d.record_success("click@e5")
        d.record_success("click@e5")
        d.record_success("scroll_down")  # Breaks the streak
        d.record_success("click@e5")
        assert d.check() is None


# --- Response Parsing Tests ---


class TestParseEscalationResponse:
    def test_click_action(self):
        raw = """Thought: I see a search button at the top right.
Action: click(start_box='(452,128)')"""
        result = _parse_escalation_response(raw, 1280, 720)
        assert result.success is True
        assert result.action_type == "click"
        assert result.x == round(452 / 1000 * 1280)
        assert result.y == round(128 / 1000 * 720)
        assert "search button" in result.thought

    def test_scroll_action(self):
        raw = """Thought: Need to scroll down to see more content.
Action: scroll(start_box='(500,500)', direction='down')"""
        result = _parse_escalation_response(raw, 1280, 720)
        assert result.success is True
        assert result.action_type == "scroll"
        assert result.direction == "down"

    def test_type_action(self):
        raw = """Thought: I need to type the search query.
Action: type(content='hello world')"""
        result = _parse_escalation_response(raw, 1280, 720)
        assert result.success is True
        assert result.action_type == "type"
        assert result.text == "hello world"

    def test_finished_maps_to_done(self):
        raw = """Thought: Task is complete.
Action: finished(content='done')"""
        result = _parse_escalation_response(raw, 1280, 720)
        assert result.action_type == "done"

    def test_wait_action(self):
        raw = """Thought: Page is loading.
Action: wait()"""
        result = _parse_escalation_response(raw, 1280, 720)
        assert result.action_type == "wait"

    def test_no_action_raises(self):
        with pytest.raises(ValueError, match="No Action: found"):
            _parse_escalation_response("Just some random text", 1280, 720)

    def test_box_start_format(self):
        raw = "Thought: Click.\nAction: click(start_box='<|box_start|>(300,400)<|box_end|>')"
        result = _parse_escalation_response(raw, 1280, 720)
        assert result.action_type == "click"
        assert result.x == round(300 / 1000 * 1280)
        assert result.y == round(400 / 1000 * 720)

    def test_coordinates_clamped(self):
        raw = "Thought: Edge.\nAction: click(start_box='(1100,1100)')"
        result = _parse_escalation_response(raw, 1280, 720)
        # 1100/1000 * 1280 = 1408 → clamped to 1280
        assert result.x == 1280
        assert result.y == 720

    def test_no_thought_still_parses(self):
        raw = "Action: click(start_box='(500,500)')"
        result = _parse_escalation_response(raw, 1280, 720)
        assert result.success is True
        assert result.thought == ""


# --- VisionEscalation Integration Tests ---


class TestVisionEscalation:
    async def test_escalate_success(self):
        """Full escalation flow: screenshot → infer → parse → result."""
        ollama = MagicMock()
        ollama.chat = AsyncMock(return_value={
            "message": {
                "content": "Thought: I see a search box.\nAction: click(start_box='(500,300)')"
            }
        })

        escalation = VisionEscalation(
            ollama_client=ollama,
            model="test-model",
            screen_width=1280,
            screen_height=720,
        )

        page = AsyncMock()
        page.screenshot = AsyncMock(return_value=b"fake-jpeg-bytes")

        result = await escalation.escalate(page, "Search for cats")

        assert result.success is True
        assert result.action_type == "click"
        assert result.x is not None
        assert result.y is not None
        page.screenshot.assert_called_once_with(type="jpeg", quality=85)
        ollama.chat.assert_called_once()

    async def test_escalate_screenshot_failure(self):
        ollama = MagicMock()
        escalation = VisionEscalation(
            ollama_client=ollama,
            model="test-model",
        )

        page = AsyncMock()
        page.screenshot = AsyncMock(side_effect=Exception("browser crashed"))

        result = await escalation.escalate(page, "Do something")

        assert result.success is False
        assert "Screenshot failed" in result.error

    async def test_escalate_inference_failure(self):
        ollama = MagicMock()
        ollama.chat = AsyncMock(side_effect=Exception("model not found"))

        escalation = VisionEscalation(
            ollama_client=ollama,
            model="nonexistent-model",
        )

        page = AsyncMock()
        page.screenshot = AsyncMock(return_value=b"fake-bytes")

        result = await escalation.escalate(page, "Do something")

        assert result.success is False
        assert "Inference failed" in result.error

    async def test_escalate_parse_failure(self):
        ollama = MagicMock()
        ollama.chat = AsyncMock(return_value={
            "message": {"content": "I don't understand the task"}
        })

        escalation = VisionEscalation(
            ollama_client=ollama,
            model="test-model",
        )

        page = AsyncMock()
        page.screenshot = AsyncMock(return_value=b"fake-bytes")

        result = await escalation.escalate(page, "Do something")

        assert result.success is False
        assert "Parse error" in result.error

    async def test_escalate_includes_context(self):
        """Verify recent actions are passed to the vision model."""
        ollama = MagicMock()
        ollama.chat = AsyncMock(return_value={
            "message": {
                "content": "Thought: Clicking.\nAction: click(start_box='(100,200)')"
            }
        })

        escalation = VisionEscalation(
            ollama_client=ollama,
            model="test-model",
        )

        page = AsyncMock()
        page.screenshot = AsyncMock(return_value=b"fake-bytes")

        await escalation.escalate(
            page, "Search for cats",
            recent_actions=["Step 1: click @e5 → OK", "Step 2: type @e3 → FAILED"],
            reason=StuckReason.PARSE_FAILURES,
        )

        # Check that the prompt includes context
        call_args = ollama.chat.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages") or call_args[0][1]
        prompt_text = messages[0]["content"]
        assert "parse_failures" in prompt_text
        assert "Step 1" in prompt_text
