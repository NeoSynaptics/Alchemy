"""Tests for VoiceAgent — voice pipeline control agent."""

from unittest.mock import AsyncMock, MagicMock, PropertyMock
from dataclasses import dataclass

import pytest

from alchemy.connect.agents.voice_agent import VoiceAgent
from alchemy.connect.protocol import AlchemyMessage


@dataclass
class FakeVoiceStatus:
    state: str = "idle"
    mode: str = "conversation"
    is_running: bool = True

    def to_dict(self):
        return {"state": self.state, "mode": self.mode, "is_running": self.is_running}


class FakeVoiceSystem:
    def __init__(self, running=True):
        self.is_running = running
        self._mode = MagicMock()
        self._mode.value = "conversation"

    @property
    def mode(self):
        return self._mode

    def status(self):
        return FakeVoiceStatus(is_running=self.is_running)

    async def start(self):
        self.is_running = True

    async def stop(self):
        self.is_running = False

    def set_mode(self, mode):
        self._mode = mode


class FakeAppState:
    def __init__(self, voice_system=None):
        self.voice_system = voice_system


@pytest.fixture
def agent_no_voice():
    return VoiceAgent(FakeAppState(voice_system=None))


@pytest.fixture
def agent_with_voice():
    return VoiceAgent(FakeAppState(voice_system=FakeVoiceSystem()))


@pytest.fixture
def agent_stopped_voice():
    return VoiceAgent(FakeAppState(voice_system=FakeVoiceSystem(running=False)))


class TestVoiceAgent:
    def test_agent_id(self, agent_no_voice):
        assert agent_no_voice.agent_id == "voice"

    def test_describe_no_voice(self, agent_no_voice):
        desc = agent_no_voice.describe()
        assert desc["available"] is False

    def test_describe_with_voice(self, agent_with_voice):
        desc = agent_with_voice.describe()
        assert desc["available"] is True
        assert "start" in desc["types"]
        assert "stop" in desc["types"]
        assert "mode" in desc["types"]
        assert "say" in desc["types"]

    @pytest.mark.asyncio
    async def test_status_no_voice(self, agent_no_voice):
        msg = AlchemyMessage(agent="voice", type="status")
        responses = [r async for r in agent_no_voice.handle(msg, "dev1")]
        assert len(responses) == 1
        assert responses[0].payload["running"] is False
        assert responses[0].payload["available"] is False

    @pytest.mark.asyncio
    async def test_status_with_voice(self, agent_with_voice):
        msg = AlchemyMessage(agent="voice", type="status")
        responses = [r async for r in agent_with_voice.handle(msg, "dev1")]
        assert len(responses) == 1
        assert responses[0].type == "status"
        assert responses[0].payload["available"] is True
        assert responses[0].payload["is_running"] is True

    @pytest.mark.asyncio
    async def test_start_no_voice(self, agent_no_voice):
        msg = AlchemyMessage(agent="voice", type="start")
        responses = [r async for r in agent_no_voice.handle(msg, "dev1")]
        assert responses[0].type == "error"

    @pytest.mark.asyncio
    async def test_start_already_running(self, agent_with_voice):
        msg = AlchemyMessage(agent="voice", type="start")
        responses = [r async for r in agent_with_voice.handle(msg, "dev1")]
        assert responses[0].type == "started"
        assert responses[0].payload["status"] == "already_running"

    @pytest.mark.asyncio
    async def test_start_success(self, agent_stopped_voice):
        msg = AlchemyMessage(agent="voice", type="start")
        responses = [r async for r in agent_stopped_voice.handle(msg, "dev1")]
        assert responses[0].type == "started"
        assert responses[0].payload["status"] == "started"

    @pytest.mark.asyncio
    async def test_stop_no_voice(self, agent_no_voice):
        msg = AlchemyMessage(agent="voice", type="stop")
        responses = [r async for r in agent_no_voice.handle(msg, "dev1")]
        assert responses[0].type == "error"

    @pytest.mark.asyncio
    async def test_stop_already_stopped(self, agent_stopped_voice):
        msg = AlchemyMessage(agent="voice", type="stop")
        responses = [r async for r in agent_stopped_voice.handle(msg, "dev1")]
        assert responses[0].type == "stopped"
        assert responses[0].payload["status"] == "already_stopped"

    @pytest.mark.asyncio
    async def test_stop_success(self, agent_with_voice):
        msg = AlchemyMessage(agent="voice", type="stop")
        responses = [r async for r in agent_with_voice.handle(msg, "dev1")]
        assert responses[0].type == "stopped"
        assert responses[0].payload["status"] == "stopped"

    @pytest.mark.asyncio
    async def test_mode_invalid(self, agent_with_voice):
        msg = AlchemyMessage(agent="voice", type="mode", payload={"mode": "bogus"})
        responses = [r async for r in agent_with_voice.handle(msg, "dev1")]
        assert responses[0].type == "error"
        assert "Invalid mode" in responses[0].payload["reason"]

    @pytest.mark.asyncio
    async def test_mode_no_voice(self, agent_no_voice):
        msg = AlchemyMessage(agent="voice", type="mode", payload={"mode": "command"})
        responses = [r async for r in agent_no_voice.handle(msg, "dev1")]
        assert responses[0].type == "error"

    @pytest.mark.asyncio
    async def test_mode_change_success(self, agent_with_voice):
        msg = AlchemyMessage(agent="voice", type="mode", payload={"mode": "conversation"})
        responses = [r async for r in agent_with_voice.handle(msg, "dev1")]
        assert responses[0].type == "mode_changed"
        assert responses[0].payload["mode"] == "conversation"

    @pytest.mark.asyncio
    async def test_say_no_voice(self, agent_no_voice):
        msg = AlchemyMessage(agent="voice", type="say", payload={"text": "hello"})
        responses = [r async for r in agent_no_voice.handle(msg, "dev1")]
        assert responses[0].type == "error"

    @pytest.mark.asyncio
    async def test_say_empty_text(self, agent_with_voice):
        msg = AlchemyMessage(agent="voice", type="say", payload={"text": ""})
        responses = [r async for r in agent_with_voice.handle(msg, "dev1")]
        assert responses[0].type == "error"
        assert "Empty text" in responses[0].payload["reason"]

    @pytest.mark.asyncio
    async def test_say_no_pipeline(self, agent_with_voice):
        msg = AlchemyMessage(agent="voice", type="say", payload={"text": "hello world"})
        responses = [r async for r in agent_with_voice.handle(msg, "dev1")]
        # No _pipeline on FakeVoiceSystem, so falls through to "TTS not accessible"
        assert responses[0].type == "error"
        assert "TTS not accessible" in responses[0].payload["reason"]

    @pytest.mark.asyncio
    async def test_unknown_type(self, agent_with_voice):
        msg = AlchemyMessage(agent="voice", type="bogus")
        responses = [r async for r in agent_with_voice.handle(msg, "dev1")]
        assert responses[0].type == "error"
        assert "Unknown type" in responses[0].payload["reason"]
