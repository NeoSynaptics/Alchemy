"""Tests for AlchemyConnect agent router."""

import pytest

from alchemy.connect.protocol import AlchemyMessage
from alchemy.connect.router import AgentRouter, ConnectAgent


class EchoAgent(ConnectAgent):
    """Test agent that echoes back messages."""

    @property
    def agent_id(self) -> str:
        return "echo"

    async def handle(self, msg, device_id):
        yield AlchemyMessage(
            agent="echo",
            type="echo",
            payload={"text": msg.payload.get("text", ""), "from": device_id},
            ref=msg.id,
        )


class MultiAgent(ConnectAgent):
    """Test agent that yields multiple responses."""

    @property
    def agent_id(self) -> str:
        return "multi"

    async def handle(self, msg, device_id):
        for i in range(3):
            yield AlchemyMessage(
                agent="multi", type="chunk", payload={"i": i}, ref=msg.id,
            )


class TestAgentRouter:
    def test_register_and_get(self):
        router = AgentRouter()
        agent = EchoAgent()
        router.register(agent)
        assert router.get("echo") is agent

    def test_get_unknown(self):
        router = AgentRouter()
        assert router.get("nonexistent") is None

    def test_available_agents(self):
        router = AgentRouter()
        router.register(EchoAgent())
        router.register(MultiAgent())
        assert sorted(router.available_agents) == ["echo", "multi"]

    def test_describe_all(self):
        router = AgentRouter()
        router.register(EchoAgent())
        descriptions = router.describe_all()
        assert len(descriptions) == 1
        assert descriptions[0]["agent_id"] == "echo"

    def test_overwrite_agent(self):
        router = AgentRouter()
        a1 = EchoAgent()
        a2 = EchoAgent()
        router.register(a1)
        router.register(a2)
        assert router.get("echo") is a2
        assert len(router.available_agents) == 1


class TestConnectAgent:
    @pytest.mark.asyncio
    async def test_echo_agent(self):
        agent = EchoAgent()
        msg = AlchemyMessage(agent="echo", type="msg", payload={"text": "hello"})
        responses = [r async for r in agent.handle(msg, "dev1")]
        assert len(responses) == 1
        assert responses[0].payload["text"] == "hello"
        assert responses[0].payload["from"] == "dev1"
        assert responses[0].ref == msg.id

    @pytest.mark.asyncio
    async def test_multi_agent(self):
        agent = MultiAgent()
        msg = AlchemyMessage(agent="multi", type="go")
        responses = [r async for r in agent.handle(msg, "dev1")]
        assert len(responses) == 3
        assert [r.payload["i"] for r in responses] == [0, 1, 2]

    def test_describe_default(self):
        agent = EchoAgent()
        desc = agent.describe()
        assert desc["agent_id"] == "echo"
