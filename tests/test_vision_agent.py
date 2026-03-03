"""Vision agent tests — full agent loop with all deps mocked."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from alchemy.agent.task_manager import TaskManager
from alchemy.agent.vision_agent import VisionAgent
from alchemy.models.ollama_client import OllamaClient
from alchemy.schemas import TaskStatus


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
        history_window=4, screen_width=1920, screen_height=1080,
        use_streaming=True, model_routing=False,
        temperature=0.0, max_tokens=384,
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

        # Agent uses chat_stream (returns string directly)
        ollama.chat_stream = AsyncMock(side_effect=[
            "Thought: Click it.\nAction: click(start_box='(500,500)')",
            "Thought: Done.\nAction: finished(content='Complete')",
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

        ollama.chat_stream = AsyncMock(return_value=(
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

        ollama.chat_stream = AsyncMock(side_effect=Exception("Connection refused"))

        agent = _make_agent(*deps)
        status = await agent.run_task(tid, "inference fail")

        assert status == TaskStatus.FAILED
        assert "Inference error" in tm.get_task(tid).error

    async def test_parse_error_continues(self, deps):
        ollama, controller, neotx, tm = deps
        tid = uuid4()
        tm.create_task(tid, "parse error")

        # Step 1: garbled, Step 2: valid finished
        ollama.chat_stream = AsyncMock(side_effect=[
            "I don't know what to do",
            "Thought: Done.\nAction: finished(content='ok')",
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

    async def test_non_streaming_mode(self, deps):
        """When streaming is disabled, use chat() instead."""
        ollama, controller, neotx, tm = deps
        tid = uuid4()
        tm.create_task(tid, "non-streaming")

        ollama.chat = AsyncMock(return_value={
            "message": {"role": "assistant", "content": "Thought: Done.\nAction: finished(content='ok')"},
        })

        agent = _make_agent(*deps, use_streaming=False)
        status = await agent.run_task(tid, "non-streaming")

        assert status == TaskStatus.COMPLETED
        ollama.chat.assert_called()

    async def test_model_escalation_on_failure(self, deps):
        """When fast model fails, escalate to full model."""
        ollama, controller, neotx, tm = deps
        tid = uuid4()
        tm.create_task(tid, "escalation test")

        call_count = 0

        async def mock_stream(model, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if model == "fast-model" and call_count == 1:
                raise Exception("Fast model OOM")
            return "Thought: Done.\nAction: finished(content='ok')"

        ollama.chat_stream = AsyncMock(side_effect=mock_stream)

        agent = _make_agent(*deps, fast_model="fast-model", model_routing=True)
        status = await agent.run_task(tid, "open spotify and play music")

        assert status == TaskStatus.COMPLETED


class TestAnalyzeSingle:
    async def test_returns_action(self, deps):
        ollama, controller, neotx, tm = deps
        ollama.chat_stream = AsyncMock(return_value=(
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

    async def test_non_streaming_analyze(self, deps):
        ollama, controller, neotx, tm = deps
        ollama.chat = AsyncMock(return_value={
            "message": {"role": "assistant", "content": "Thought: Click.\nAction: click(start_box='(500,500)')"},
        })

        agent = _make_agent(*deps, use_streaming=False)
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        resp = await agent.analyze_single(png, "click it")

        assert resp.action.action == "click"
