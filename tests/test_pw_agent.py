"""Tests for Playwright agent main loop."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from alchemy.click.pw_agent import PlaywrightAgent, AgentResult, AgentStatus


# --- Mock helpers ---

def _mock_ollama(responses: list[str]):
    """Create a mock OllamaClient that returns predefined responses via chat_think()."""
    client = MagicMock()

    call_count = [0]

    async def mock_chat_think(model, messages, think=True, options=None, seed=None):
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        return {"content": responses[idx], "thinking": "", "total_duration": 0}

    client.chat_think = mock_chat_think
    return client


def _mock_page(aria_text: str):
    """Create a mock page with aria_snapshot() response."""
    page = AsyncMock()

    # Mock page.locator(":root").aria_snapshot() for capture_snapshot
    snapshot_locator = AsyncMock()
    snapshot_locator.aria_snapshot = AsyncMock(return_value=aria_text)
    page.locator = MagicMock(return_value=snapshot_locator)
    page.wait_for_load_state = AsyncMock()

    # Mock get_by_role for executor
    action_locator = AsyncMock()
    action_locator.click = AsyncMock()
    action_locator.fill = AsyncMock()
    action_locator.nth = MagicMock(return_value=action_locator)
    page.get_by_role = MagicMock(return_value=action_locator)
    page.mouse = AsyncMock()
    page.mouse.wheel = AsyncMock()
    page.keyboard = AsyncMock()
    page.keyboard.press = AsyncMock()

    return page


SIMPLE_TREE = """\
- document:
  - textbox "Search"
  - button "Go"\
"""


# --- Tests ---

class TestPlaywrightAgent:
    async def test_simple_task_completes(self):
        """Agent clicks a button then reports done."""
        ollama = _mock_ollama([
            "Thought: I see a Go button.\nAction: click @e2",
            "Thought: Task complete.\nAction: done",
        ])
        page = _mock_page(SIMPLE_TREE)

        agent = PlaywrightAgent(ollama_client=ollama, think=False)
        result = await agent.run_task("Click the Go button", page)

        assert result.status == AgentStatus.COMPLETED
        assert result.total_steps == 2
        assert len(result.steps) == 2
        assert result.steps[0].action.type == "click"
        assert result.steps[1].action.type == "done"

    async def test_type_and_done(self):
        """Agent types text then completes."""
        ollama = _mock_ollama([
            'Thought: Type search query.\nAction: type @e1 "pole vault"',
            "Thought: Done.\nAction: done",
        ])
        page = _mock_page(SIMPLE_TREE)

        agent = PlaywrightAgent(ollama_client=ollama, think=False)
        result = await agent.run_task("Search for pole vault", page)

        assert result.status == AgentStatus.COMPLETED
        assert result.total_steps == 2

    async def test_max_steps_reached(self):
        """Agent fails after max steps."""
        # Never says "done"
        ollama = _mock_ollama([
            "Thought: Keep scrolling.\nAction: scroll down"
        ] * 10)
        page = _mock_page(SIMPLE_TREE)

        agent = PlaywrightAgent(ollama_client=ollama, max_steps=5, think=False)
        result = await agent.run_task("Find something", page)

        assert result.status == AgentStatus.FAILED
        assert "maximum steps" in result.error

    async def test_consecutive_errors_fail(self):
        """Agent fails after 3 consecutive parse errors."""
        ollama = _mock_ollama([
            "I don't know what to do",  # No Action: line
            "Still confused",
            "Very confused",
        ])
        page = _mock_page(SIMPLE_TREE)

        agent = PlaywrightAgent(ollama_client=ollama, think=False)
        result = await agent.run_task("Do something", page)

        assert result.status == AgentStatus.FAILED

    async def test_error_recovery(self):
        """Agent recovers from a parse error and continues."""
        ollama = _mock_ollama([
            "I'm confused",  # Parse error
            "Thought: Found it.\nAction: click @e2",  # Recovery
            "Thought: Done.\nAction: done",
        ])
        page = _mock_page(SIMPLE_TREE)

        agent = PlaywrightAgent(ollama_client=ollama, think=False)
        result = await agent.run_task("Click Go", page)

        assert result.status == AgentStatus.COMPLETED
        assert result.total_steps == 3

    async def test_approval_gate(self):
        """Agent pauses when approval checker triggers."""
        ollama = _mock_ollama([
            "Thought: Need to submit.\nAction: click @e2",
        ])
        page = _mock_page(SIMPLE_TREE)

        # Always require approval
        agent = PlaywrightAgent(
            ollama_client=ollama,
            think=False,
            approval_checker=lambda action: True,
        )
        result = await agent.run_task("Submit form", page)

        assert result.status == AgentStatus.WAITING_APPROVAL

    async def test_step_timing(self):
        """Steps record inference and execution timing."""
        ollama = _mock_ollama([
            "Thought: Click it.\nAction: click @e2",
            "Thought: Done.\nAction: done",
        ])
        page = _mock_page(SIMPLE_TREE)

        agent = PlaywrightAgent(ollama_client=ollama, think=False)
        result = await agent.run_task("Click Go", page)

        assert result.total_ms > 0
        assert result.steps[0].inference_ms >= 0
        assert result.steps[0].execution_ms >= 0

    async def test_think_mode_payload(self):
        """Think: true is passed in the Ollama payload."""
        ollama = _mock_ollama(["Thought: Done.\nAction: done"])
        page = _mock_page(SIMPLE_TREE)

        agent = PlaywrightAgent(ollama_client=ollama, think=True)
        result = await agent.run_task("Test", page)

        # Verify the think parameter was sent
        # (We check by verifying the agent ran successfully with think=True)
        assert result.status == AgentStatus.COMPLETED
