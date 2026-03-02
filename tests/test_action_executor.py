"""Action executor tests — mock ShadowDesktopController."""

from unittest.mock import AsyncMock

import pytest

from alchemy.agent.action_executor import ActionExecutor
from alchemy.schemas import VisionAction


@pytest.fixture
def executor():
    mock_controller = AsyncMock()
    mock_controller.execute = AsyncMock(return_value="")
    return ActionExecutor(mock_controller)


class TestExecute:
    async def test_click(self, executor):
        action = VisionAction(action="click", x=100, y=200)
        await executor.execute(action)
        cmd = executor._controller.execute.call_args[0][0]
        assert "mousemove --sync 100 200" in cmd
        assert "click 1" in cmd

    async def test_double_click(self, executor):
        action = VisionAction(action="double_click", x=100, y=200)
        await executor.execute(action)
        cmd = executor._controller.execute.call_args[0][0]
        assert "--repeat 2" in cmd

    async def test_right_click(self, executor):
        action = VisionAction(action="right_click", x=100, y=200)
        await executor.execute(action)
        cmd = executor._controller.execute.call_args[0][0]
        assert "click 3" in cmd

    async def test_drag(self, executor):
        action = VisionAction(action="drag", x=100, y=200, end_x=300, end_y=400)
        await executor.execute(action)
        cmd = executor._controller.execute.call_args[0][0]
        assert "mousedown 1" in cmd
        assert "mouseup 1" in cmd

    async def test_type(self, executor):
        action = VisionAction(action="type", text="hello world")
        await executor.execute(action)
        cmd = executor._controller.execute.call_args[0][0]
        assert "xdotool type" in cmd
        assert "hello world" in cmd

    async def test_hotkey(self, executor):
        action = VisionAction(action="hotkey", text="ctrl+c")
        await executor.execute(action)
        cmd = executor._controller.execute.call_args[0][0]
        assert "xdotool key" in cmd
        assert "ctrl+c" in cmd

    async def test_scroll_down(self, executor):
        action = VisionAction(action="scroll", x=500, y=500, direction="down", amount=3)
        await executor.execute(action)
        cmd = executor._controller.execute.call_args[0][0]
        assert "click --repeat 3" in cmd
        assert "5" in cmd  # button 5 = scroll down

    async def test_scroll_up(self, executor):
        action = VisionAction(action="scroll", x=500, y=500, direction="up", amount=2)
        await executor.execute(action)
        cmd = executor._controller.execute.call_args[0][0]
        assert "4" in cmd  # button 4 = scroll up

    async def test_wait(self, executor):
        action = VisionAction(action="wait")
        result = await executor.execute(action)
        assert result == "waited"
        executor._controller.execute.assert_not_called()

    async def test_done_no_execute(self, executor):
        action = VisionAction(action="done")
        result = await executor.execute(action)
        assert result == ""
        executor._controller.execute.assert_not_called()

    async def test_fail_no_execute(self, executor):
        action = VisionAction(action="fail")
        result = await executor.execute(action)
        assert result == ""

    async def test_unknown_raises(self, executor):
        action = VisionAction(action="unknown_action")
        with pytest.raises(ValueError, match="Unknown action"):
            await executor.execute(action)
