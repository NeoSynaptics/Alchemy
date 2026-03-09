"""Tests for ScreenshotSummarizer — VLM screenshot captioning."""

from unittest.mock import AsyncMock

import pytest

from alchemy.memory.timeline.summarizer import ScreenshotSummarizer


@pytest.fixture
def mock_ollama():
    client = AsyncMock()
    client.chat = AsyncMock(return_value={
        "message": {"content": "User is editing Python code in VS Code"},
        "total_duration": 3000000000,
    })
    return client


@pytest.mark.asyncio
async def test_summarize_returns_caption(mock_ollama):
    summarizer = ScreenshotSummarizer(mock_ollama, model="qwen2.5vl:7b")
    result = await summarizer.summarize(b"\xff\xd8fake_jpeg")

    assert result == "User is editing Python code in VS Code"
    mock_ollama.chat.assert_awaited_once()

    call_kwargs = mock_ollama.chat.call_args
    assert call_kwargs.kwargs["model"] == "qwen2.5vl:7b"
    assert call_kwargs.kwargs["images"] == [b"\xff\xd8fake_jpeg"]
    assert call_kwargs.kwargs["options"]["num_ctx"] == 8192


@pytest.mark.asyncio
async def test_summarize_returns_empty_on_failure(mock_ollama):
    mock_ollama.chat.side_effect = RuntimeError("VLM timeout")
    summarizer = ScreenshotSummarizer(mock_ollama, model="qwen2.5vl:7b")

    result = await summarizer.summarize(b"fake")
    assert result == ""
