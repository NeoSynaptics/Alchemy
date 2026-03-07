"""Tests for ImageAgent — stub image generation agent."""

import pytest

from alchemy.connect.agents.image_agent import ImageAgent
from alchemy.connect.protocol import AlchemyMessage


class FakeAppState:
    pass


@pytest.fixture
def agent():
    return ImageAgent(FakeAppState())


class TestImageAgent:
    def test_agent_id(self, agent):
        assert agent.agent_id == "image"

    def test_describe(self, agent):
        desc = agent.describe()
        assert desc["agent_id"] == "image"
        assert desc["available"] is False
        assert "generate" in desc["types"]
        assert "status" in desc["types"]

    @pytest.mark.asyncio
    async def test_status_not_available(self, agent):
        msg = AlchemyMessage(agent="image", type="status")
        responses = [r async for r in agent.handle(msg, "dev1")]
        assert len(responses) == 1
        assert responses[0].type == "status"
        assert responses[0].payload["available"] is False

    @pytest.mark.asyncio
    async def test_generate_not_available(self, agent):
        msg = AlchemyMessage(
            agent="image", type="generate",
            payload={"prompt": "sunset", "width": 512, "height": 512},
        )
        responses = [r async for r in agent.handle(msg, "dev1")]
        assert len(responses) == 1
        assert responses[0].type == "error"
        assert "not available" in responses[0].payload["reason"]

    @pytest.mark.asyncio
    async def test_unknown_type(self, agent):
        msg = AlchemyMessage(agent="image", type="bogus")
        responses = [r async for r in agent.handle(msg, "dev1")]
        assert len(responses) == 1
        assert responses[0].type == "error"
        assert "Unknown type" in responses[0].payload["reason"]

    def test_gpu_guard_accepted(self):
        """ImageAgent accepts a gpu_guard parameter."""
        import asyncio
        sem = asyncio.Semaphore(2)
        agent = ImageAgent(FakeAppState(), gpu_guard=sem)
        assert agent._gpu_guard is sem
