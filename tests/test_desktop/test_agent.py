"""Tests for DesktopAgent — vision-driven desktop automation loop."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alchemy.desktop.agent import (
    DesktopAgent,
    DesktopTaskStatus,
    _parse_response,
)
from alchemy.desktop.controller import DesktopController, ScreenInfo


# --- Response Parsing Tests ---


class TestParseResponse:
    def test_click_action(self):
        raw = 'Thought: I see a button.\nAction: click(start_box="(450,200)")'
        action, x, y, text, direction, thought = _parse_response(raw)
        assert action == "click"
        assert x == 450
        assert y == 200
        assert text is None
        assert thought == "I see a button."

    def test_type_action(self):
        raw = 'Thought: Input field is focused.\nAction: type(content="hello world")'
        action, x, y, text, direction, thought = _parse_response(raw)
        assert action == "type"
        assert text == "hello world"
        assert x is None

    def test_scroll_action(self):
        raw = 'Thought: Need to scroll.\nAction: scroll(start_box="(640,360)", direction="down")'
        action, x, y, text, direction, thought = _parse_response(raw)
        assert action == "scroll"
        assert x == 640
        assert y == 360
        assert direction == "down"

    def test_finished_action(self):
        raw = 'Thought: Task complete.\nAction: finished(content="done")'
        action, x, y, text, direction, thought = _parse_response(raw)
        assert action == "done"
        assert text == "done"

    def test_hotkey_action(self):
        raw = 'Thought: Copy text.\nAction: hotkey(key="ctrl+c")'
        action, x, y, text, direction, thought = _parse_response(raw)
        assert action == "hotkey"
        assert text == "ctrl+c"

    def test_wait_action(self):
        raw = "Thought: Page loading.\nAction: wait()"
        action, x, y, text, direction, thought = _parse_response(raw)
        assert action == "wait"

    def test_no_thought(self):
        raw = 'Action: click(start_box="(100,200)")'
        action, x, y, text, direction, thought = _parse_response(raw)
        assert action == "click"
        assert x == 100
        assert y == 200
        assert thought == ""

    def test_no_action_raises(self):
        with pytest.raises(ValueError, match="No Action:"):
            _parse_response("I don't know what to do")

    def test_double_click(self):
        raw = 'Thought: Double click file.\nAction: left_double(start_box="(300,400)")'
        action, x, y, text, direction, thought = _parse_response(raw)
        assert action == "double_click"
        assert x == 300
        assert y == 400

    def test_right_click(self):
        raw = 'Thought: Context menu.\nAction: right_single(start_box="(300,400)")'
        action, x, y, text, direction, thought = _parse_response(raw)
        assert action == "right_click"

    def test_type_with_single_quotes(self):
        raw = "Thought: Typing.\nAction: type(content='hello')"
        action, x, y, text, direction, thought = _parse_response(raw)
        assert action == "type"
        assert text == "hello"

    def test_multiline_thought(self):
        raw = (
            "Thought: I see a blue button in the top right.\n"
            "It looks like a submit button.\n"
            "Action: click(start_box=\"(900,50)\")"
        )
        action, x, y, text, direction, thought = _parse_response(raw)
        assert action == "click"
        assert x == 900
        assert y == 50
        assert "blue button" in thought

    def test_click_at_format(self):
        """minicpm-v sometimes outputs click@(X,Y) format."""
        raw = "Thought: Click the button.\nAction: click@(463,158)"
        action, x, y, text, direction, thought = _parse_response(raw)
        assert action == "click"
        assert x == 463
        assert y == 158
        assert thought == "Click the button."

    def test_click_at_format_with_box_suffix(self):
        raw = "Thought: Click start.\nAction: click@(436,15</box>"
        action, x, y, text, direction, thought = _parse_response(raw)
        assert action == "click"
        assert x == 436
        assert y == 15

    def test_click_at_no_thought(self):
        raw = "Action: click@(100,200)"
        action, x, y, text, direction, thought = _parse_response(raw)
        assert action == "click"
        assert x == 100
        assert y == 200
        assert thought == ""


# --- Agent Loop Tests ---


def _make_mock_controller() -> DesktopController:
    """Create a mock desktop controller."""
    ctrl = MagicMock(spec=DesktopController)
    ctrl.image_width = 1280
    ctrl.image_height = 720
    ctrl.screen = ScreenInfo(width=1920, height=1080)
    ctrl.screenshot = AsyncMock(return_value=b"\xff\xd8fake_jpeg")
    ctrl.click = AsyncMock(return_value="clicked (500, 300)")
    ctrl.double_click = AsyncMock(return_value="double_clicked (500, 300)")
    ctrl.right_click = AsyncMock(return_value="right_clicked (500, 300)")
    ctrl.type_text = AsyncMock(return_value="typed 'hello'")
    ctrl.hotkey = AsyncMock(return_value="hotkey ctrl+c")
    ctrl.scroll = AsyncMock(return_value="scrolled down 3")
    return ctrl


def _make_mock_ollama(responses: list[str]) -> MagicMock:
    """Create a mock Ollama client that returns predefined responses."""
    mock = MagicMock()
    side_effects = [
        {"message": {"content": r}} for r in responses
    ]
    mock.chat = AsyncMock(side_effect=side_effects)
    return mock


class TestDesktopAgentRun:
    @pytest.mark.asyncio
    async def test_single_click_and_done(self):
        """Agent: click → done in 2 steps."""
        ollama = _make_mock_ollama([
            'Thought: I see a button.\nAction: click(start_box="(640,360)")',
            'Thought: Task complete.\nAction: finished(content="done")',
        ])
        ctrl = _make_mock_controller()
        agent = DesktopAgent(ollama_client=ollama, controller=ctrl, max_steps=10)

        result = await agent.run("Click the OK button")

        assert result.status == DesktopTaskStatus.COMPLETED
        assert len(result.steps) == 2
        assert result.steps[0].action_type == "click"
        assert result.steps[1].action_type == "done"
        ctrl.click.assert_called_once()

    @pytest.mark.asyncio
    async def test_click_then_type(self):
        """Agent: click input → type text → done."""
        ollama = _make_mock_ollama([
            'Thought: Click search.\nAction: click(start_box="(400,200)")',
            'Thought: Type query.\nAction: type(content="hello")',
            'Thought: Done.\nAction: finished(content="done")',
        ])
        ctrl = _make_mock_controller()
        agent = DesktopAgent(ollama_client=ollama, controller=ctrl, max_steps=10)

        result = await agent.run("Search for hello")

        assert result.status == DesktopTaskStatus.COMPLETED
        assert len(result.steps) == 3
        ctrl.click.assert_called_once()
        ctrl.type_text.assert_called_once_with("hello")

    @pytest.mark.asyncio
    async def test_max_steps_reached(self):
        """Agent stops after max_steps."""
        responses = [
            f'Thought: Step {i}.\nAction: click(start_box="({i*10},100)")'
            for i in range(5)
        ]
        ollama = _make_mock_ollama(responses)
        ctrl = _make_mock_controller()
        agent = DesktopAgent(ollama_client=ollama, controller=ctrl, max_steps=3)

        result = await agent.run("Click something")

        assert result.status == DesktopTaskStatus.FAILED
        assert "Max steps" in result.error
        assert len(result.steps) == 3

    @pytest.mark.asyncio
    async def test_screenshot_failure(self):
        """Agent fails gracefully on screenshot error."""
        ollama = _make_mock_ollama([])
        ctrl = _make_mock_controller()
        ctrl.screenshot = AsyncMock(side_effect=RuntimeError("No display"))
        agent = DesktopAgent(ollama_client=ollama, controller=ctrl)

        result = await agent.run("Click something")

        assert result.status == DesktopTaskStatus.FAILED
        assert "Screenshot failed" in result.error

    @pytest.mark.asyncio
    async def test_inference_failure(self):
        """Agent fails on Ollama error."""
        ollama = MagicMock()
        ollama.chat = AsyncMock(side_effect=Exception("Model not loaded"))
        ctrl = _make_mock_controller()
        agent = DesktopAgent(ollama_client=ollama, controller=ctrl)

        result = await agent.run("Click something")

        assert result.status == DesktopTaskStatus.FAILED
        assert "Inference failed" in result.error

    @pytest.mark.asyncio
    async def test_parse_failure_continues(self):
        """Parse failure doesn't crash the agent — it continues to next step."""
        ollama = _make_mock_ollama([
            "I'm not sure what to do here.",  # Bad response
            'Thought: Now I see.\nAction: finished(content="done")',
        ])
        ctrl = _make_mock_controller()
        agent = DesktopAgent(ollama_client=ollama, controller=ctrl, max_steps=5)

        result = await agent.run("Click something")

        assert result.status == DesktopTaskStatus.COMPLETED
        # 2 steps: 1 parse_error + 1 done
        assert len(result.steps) == 2
        assert result.steps[0].action_type == "parse_error"
        assert result.steps[1].action_type == "done"

    @pytest.mark.asyncio
    async def test_coordinate_scaling(self):
        """Coordinates scale from resized image to actual screen."""
        # Model outputs (640, 360) in 1280x720 image
        # Should scale to (960, 540) on 1920x1080 screen
        ollama = _make_mock_ollama([
            'Thought: Click center.\nAction: click(start_box="(640,360)")',
            'Thought: Done.\nAction: finished(content="done")',
        ])
        ctrl = _make_mock_controller()
        agent = DesktopAgent(ollama_client=ollama, controller=ctrl, max_steps=5)

        result = await agent.run("Click center")

        assert result.status == DesktopTaskStatus.COMPLETED
        # Check the scaled coordinates
        assert result.steps[0].x == 960  # 640/1280 * 1920
        assert result.steps[0].y == 540  # 360/720 * 1080

    @pytest.mark.asyncio
    async def test_scroll_action(self):
        ollama = _make_mock_ollama([
            'Thought: Scroll down.\nAction: scroll(start_box="(640,360)", direction="down")',
            'Thought: Done.\nAction: finished(content="done")',
        ])
        ctrl = _make_mock_controller()
        agent = DesktopAgent(ollama_client=ollama, controller=ctrl, max_steps=5)

        result = await agent.run("Scroll down")

        assert result.status == DesktopTaskStatus.COMPLETED
        ctrl.scroll.assert_called_once()

    @pytest.mark.asyncio
    async def test_hotkey_action(self):
        ollama = _make_mock_ollama([
            'Thought: Copy.\nAction: hotkey(key="ctrl+c")',
            'Thought: Done.\nAction: finished(content="done")',
        ])
        ctrl = _make_mock_controller()
        agent = DesktopAgent(ollama_client=ollama, controller=ctrl, max_steps=5)

        result = await agent.run("Copy text")

        assert result.status == DesktopTaskStatus.COMPLETED
        ctrl.hotkey.assert_called_once_with("ctrl", "c")
