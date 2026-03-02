"""Vision agent tests — full agent loop with all deps mocked."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from alchemy.agent.task_manager import TaskManager
from alchemy.agent.vision_agent import VisionAgent
from alchemy.models.ollama_client import OllamaClient
from alchemy.schemas import TaskStatus


def _make_ollama_response(text: str) -> dict:
    return {"message": {"role": "assistant", "content": text}}


@pytest.fixture
def deps():
    """Create mocked dependencies for VisionAgent."""
    ollama = AsyncMock(spec=OllamaClient)
    controller = AsyncMock()
    controller.screenshot = AsyncMock(return_value=b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    controller.execute = AsyncMock(return_value="")
    neotx = AsyncMock()
    neotx.request_approval = AsyncMock()
    neotx.notify = AsyncMock()
    neotx.task_update = AsyncMock()
    task_manager = TaskManager()
    return ollama, controller, neotx, task_manager


def _make_agent(ollama, controller, neotx, task_manager, **kwargs):
    defaults = dict(
        model="test-model", max_steps=10, timeout=30.0,
        screenshot_interval=0.0, approval_timeout=5.0,
        history_window=8, screen_width=1920, screen_height=1080,
    )
    defaults.update(kwargs)
    return VisionAgent(
        ollama=ollama, controller=controller, neotx=neotx,
        task_manager=task_manager, **defaults,
    )


class TestRunTask:
    async def test_happy_path_finishes(self, deps):
        ollama, controller, neotx, tm = deps
        tid = uuid4()
        tm.create_task(tid, "test task")

        # Step 1: click, Step 2: finished
        ollama.chat = AsyncMock(side_effect=[
            _make_ollama_response("Thought: Click it.\nAction: click(start_box='(500,500)')"),
            _make_ollama_response("Thought: Done.\nAction: finished(content='Complete')"),
        ])

        agent = _make_agent(*deps)
        status = await agent.run_task(tid, "test task")

        assert status == TaskStatus.COMPLETED
        assert tm.get_task(tid).status == TaskStatus.COMPLETED
        assert controller.execute.call_count == 1  # only click, not finished

    async def test_max_steps_exceeded(self, deps):
        ollama, controller, neotx, tm = deps
        tid = uuid4()
        tm.create_task(tid, "endless task")

        # Always returns click — never finishes
        ollama.chat = AsyncMock(return_value=_make_ollama_response(
            "Thought: Keep clicking.\nAction: click(start_box='(100,100)')"
        ))

        agent = _make_agent(*deps, max_steps=3)
        status = await agent.run_task(tid, "endless task")

        assert status == TaskStatus.FAILED
        assert "max steps" in tm.get_task(tid).error.lower()

    async def test_screenshot_failure(self, deps):
        ollama, controller, neotx, tm = deps
        tid = uuid4()
        tm.create_task(tid, "screenshot fail")

        controller.screenshot = AsyncMock(side_effect=RuntimeError("No display"))

        agent = _make_agent(*deps)
        status = await agent.run_task(tid, "screenshot fail")

        assert status == TaskStatus.FAILED
        assert "No display" in tm.get_task(tid).error

    async def test_ollama_error(self, deps):
        ollama, controller, neotx, tm = deps
        tid = uuid4()
        tm.create_task(tid, "inference fail")

        ollama.chat = AsyncMock(side_effect=Exception("Connection refused"))

        agent = _make_agent(*deps)
        status = await agent.run_task(tid, "inference fail")

        assert status == TaskStatus.FAILED
        assert "Inference error" in tm.get_task(tid).error

    async def test_parse_error_continues(self, deps):
        ollama, controller, neotx, tm = deps
        tid = uuid4()
        tm.create_task(tid, "parse error")

        # Step 1: garbled, Step 2: valid finished
        ollama.chat = AsyncMock(side_effect=[
            _make_ollama_response("I don't know what to do"),
            _make_ollama_response("Thought: Done.\nAction: finished(content='ok')"),
        ])

        agent = _make_agent(*deps)
        status = await agent.run_task(tid, "parse error")

        assert status == TaskStatus.COMPLETED

    async def test_cancellation(self, deps):
        ollama, controller, neotx, tm = deps
        tid = uuid4()
        state = tm.create_task(tid, "cancel me")
        state.cancel_event.set()

        agent = _make_agent(*deps)
        status = await agent.run_task(tid, "cancel me")

        assert status == TaskStatus.DENIED


class TestAnalyzeSingle:
    async def test_returns_action(self, deps):
        ollama, controller, neotx, tm = deps
        ollama.chat = AsyncMock(return_value=_make_ollama_response(
            "Thought: Click the button.\nAction: click(start_box='(500,250)')"
        ))

        agent = _make_agent(*deps)
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        resp = await agent.analyze_single(png, "click the button")

        assert resp.action.action == "click"
        assert resp.action.x == 960
        assert resp.action.y == 270
        assert resp.inference_ms > 0
        assert resp.model == "test-model"
