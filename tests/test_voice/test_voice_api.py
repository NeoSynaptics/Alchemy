"""Tests for /voice API endpoints."""

import sys
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

# Mock webrtcvad before voice imports (broken on Python 3.12+)
if "webrtcvad" not in sys.modules:
    _mock = MagicMock()
    _mock.Vad = MagicMock(return_value=MagicMock())
    sys.modules["webrtcvad"] = _mock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from alchemy.voice.interface import VoiceMode, VoiceStatus
from alchemy.server import app


@pytest_asyncio.fixture
async def client_with_voice():
    """Client with a mocked voice system."""
    mock_system = MagicMock()
    mock_system.is_running = False
    mock_system.mode = VoiceMode.CONVERSATION
    mock_system.conversation_id = uuid4()
    mock_system._router = None
    mock_system.status.return_value = VoiceStatus(
        running=False,
        mode=VoiceMode.CONVERSATION,
        pipeline_state="idle",
        tts_engine="piper",
        wake_word="hey_neo",
        conversation_id=str(mock_system.conversation_id),
    )
    mock_system.start = AsyncMock()
    mock_system.stop = AsyncMock()
    mock_system.set_mode = MagicMock()

    app.state.voice_system = mock_system

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c, mock_system


@pytest_asyncio.fixture
async def client_no_voice():
    """Client with no voice system (voice disabled)."""
    app.state.voice_system = None

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestVoiceStatus:
    async def test_status_idle(self, client_with_voice):
        client, system = client_with_voice
        resp = await client.get("/v1/voice/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pipeline_state"] == "idle"
        assert data["running"] is False
        assert data["mode"] == "conversation"
        assert data["tts_engine"] == "piper"

    async def test_status_no_voice(self, client_no_voice):
        resp = await client_no_voice.get("/v1/voice/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["running"] is False
        assert data["mode"] == "muted"


class TestVoiceStart:
    async def test_start_pipeline(self, client_with_voice):
        client, system = client_with_voice
        resp = await client.post("/v1/voice/start")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "started"
        system.start.assert_called_once()

    async def test_start_already_running(self, client_with_voice):
        client, system = client_with_voice
        system.is_running = True
        resp = await client.post("/v1/voice/start")
        data = resp.json()
        assert data["status"] == "already_running"

    async def test_start_no_voice(self, client_no_voice):
        resp = await client_no_voice.post("/v1/voice/start")
        data = resp.json()
        assert "error" in data


class TestVoiceStop:
    async def test_stop_pipeline(self, client_with_voice):
        client, system = client_with_voice
        system.is_running = True
        resp = await client.post("/v1/voice/stop")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "stopped"
        system.stop.assert_called_once()

    async def test_stop_already_stopped(self, client_with_voice):
        client, system = client_with_voice
        system.is_running = False
        resp = await client.post("/v1/voice/stop")
        data = resp.json()
        assert data["status"] == "already_stopped"

    async def test_stop_no_voice(self, client_no_voice):
        resp = await client_no_voice.post("/v1/voice/stop")
        data = resp.json()
        assert "error" in data


class TestVoiceMode:
    async def test_set_mode(self, client_with_voice):
        client, system = client_with_voice
        resp = await client.post("/v1/voice/mode", json={"mode": "command"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "command"

    async def test_invalid_mode(self, client_with_voice):
        client, system = client_with_voice
        resp = await client.post("/v1/voice/mode", json={"mode": "nonexistent"})
        data = resp.json()
        assert "error" in data
