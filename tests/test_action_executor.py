"""Action executor tests — mock desktop controller."""

from unittest.mock import AsyncMock

import pytest

from alchemy.click.action_executor import ActionExecutor
from alchemy.schemas import VisionAction


@pytest.fixture
def executor():
    mock_controller = AsyncMock()
    mock_controller.click = AsyncMock(return_value="")
    mock_controller.double_click = AsyncMock(return_value="")
    mock_controller.right_click = AsyncMock(return_value="")
    mock_controller.type_text = AsyncMock(return_value="")
    mock_controller.hotkey = AsyncMock(return_value="")
    mock_controller.scroll = AsyncMock(return_value="")
    mock_controller.move_to = AsyncMock(return_value="")
    return ActionExecutor(mock_controller)


class TestExecute:
    async def test_click(self, executor):
        action = VisionAction(action="click", x=100, y=200)
        await executor.execute(action)
        executor._controller.click.assert_awaited_once_with(100, 200)

    async def test_double_click(self, executor):
        action = VisionAction(action="double_click", x=100, y=200)
        await executor.execute(action)
        executor._controller.double_click.assert_awaited_once_with(100, 200)

    async def test_right_click(self, executor):
        action = VisionAction(action="right_click", x=100, y=200)
        await executor.execute(action)
        executor._controller.right_click.assert_awaited_once_with(100, 200)

    async def test_drag(self, executor):
        action = VisionAction(action="drag", x=100, y=200, end_x=300, end_y=400)
        await executor.execute(action)
        executor._controller.click.assert_awaited_once_with(100, 200)
        executor._controller.move_to.assert_awaited_once_with(300, 400)

    async def test_type(self, executor):
        action = VisionAction(action="type", text="hello world")
        await executor.execute(action)
        executor._controller.type_text.assert_awaited_once_with("hello world")

    async def test_hotkey(self, executor):
        action = VisionAction(action="hotkey", text="ctrl+c")
        await executor.execute(action)
        executor._controller.hotkey.assert_awaited_once_with("ctrl+c")

    async def test_scroll_down(self, executor):
        action = VisionAction(action="scroll", x=500, y=500, direction="down", amount=3)
        await executor.execute(action)
        executor._controller.scroll.assert_awaited_once_with(500, 500, "down", 3)

    async def test_scroll_up(self, executor):
        action = VisionAction(action="scroll", x=500, y=500, direction="up", amount=2)
        await executor.execute(action)
        executor._controller.scroll.assert_awaited_once_with(500, 500, "up", 2)

    async def test_wait(self, executor):
        action = VisionAction(action="wait")
        result = await executor.execute(action)
        assert result == "waited"
        executor._controller.click.assert_not_awaited()

    async def test_done_no_execute(self, executor):
        action = VisionAction(action="done")
        result = await executor.execute(action)
        assert result == ""
        executor._controller.click.assert_not_awaited()

    async def test_fail_no_execute(self, executor):
        action = VisionAction(action="fail")
        result = await executor.execute(action)
        assert result == ""

    async def test_unknown_raises(self, executor):
        action = VisionAction(action="unknown_action")
        with pytest.raises(ValueError, match="Unknown action"):
            await executor.execute(action)
