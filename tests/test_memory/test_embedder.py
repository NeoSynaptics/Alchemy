"""Tests for EmbeddingClient — wraps nomic-embed-text via Ollama."""

from unittest.mock import AsyncMock

import pytest

from alchemy.memory.timeline.embedder import EmbeddingClient


@pytest.fixture
def mock_ollama():
    client = AsyncMock()
    client.embed = AsyncMock(return_value=[0.1] * 768)
    return client


@pytest.mark.asyncio
async def test_embed_calls_ollama(mock_ollama):
    embedder = EmbeddingClient(mock_ollama, model="nomic-embed-text")
    result = await embedder.embed("Hello world")

    assert len(result) == 768
    mock_ollama.embed.assert_awaited_once_with("nomic-embed-text", "Hello world")


@pytest.mark.asyncio
async def test_embed_truncates_long_text(mock_ollama):
    embedder = EmbeddingClient(mock_ollama, model="nomic-embed-text")
    long_text = "x" * 10000
    await embedder.embed(long_text)

    call_args = mock_ollama.embed.call_args
    sent_text = call_args[0][1]
    assert len(sent_text) == 4000  # truncated to _MAX_CHARS


@pytest.mark.asyncio
async def test_embed_raises_on_failure(mock_ollama):
    mock_ollama.embed.side_effect = RuntimeError("Model not found")
    embedder = EmbeddingClient(mock_ollama, model="nomic-embed-text")

    with pytest.raises(RuntimeError, match="Model not found"):
        await embedder.embed("test")
