"""AlchemyWord writer and API tests."""

import pytest
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from alchemy.word.writer import generate, VALID_MODES
from alchemy.word.api import router


# --- Writer unit tests ---


@pytest.mark.asyncio
async def test_generate_summarize():
    client = AsyncMock()
    client.chat = AsyncMock(return_value={"message": {"content": "Short summary."}})

    result = await generate(client, "Long text here.", mode="summarize")
    assert result == "Short summary."
    client.chat.assert_called_once()
    call_args = client.chat.call_args
    assert call_args.kwargs["messages"][0]["role"] == "system"
    assert "summarize" in call_args.kwargs["messages"][0]["content"].lower()


@pytest.mark.asyncio
async def test_generate_rewrite():
    client = AsyncMock()
    client.chat = AsyncMock(return_value={"message": {"content": "Improved text."}})

    result = await generate(client, "Bad text.", mode="rewrite")
    assert result == "Improved text."


@pytest.mark.asyncio
async def test_generate_expand():
    client = AsyncMock()
    client.chat = AsyncMock(return_value={"message": {"content": "Expanded version with details."}})

    result = await generate(client, "Brief note.", mode="expand")
    assert result == "Expanded version with details."


@pytest.mark.asyncio
async def test_generate_translate():
    client = AsyncMock()
    client.chat = AsyncMock(return_value={"message": {"content": "Texto traducido."}})

    result = await generate(client, "Translate this.", mode="translate", target_language="Spanish")
    assert result == "Texto traducido."
    call_args = client.chat.call_args
    assert "Spanish" in call_args.kwargs["messages"][0]["content"]


@pytest.mark.asyncio
async def test_generate_invalid_mode():
    client = AsyncMock()
    with pytest.raises(ValueError, match="Invalid mode"):
        await generate(client, "text", mode="invalid")


def test_valid_modes():
    assert VALID_MODES == {"summarize", "rewrite", "expand", "translate"}


# --- API tests ---


def _make_app(ollama_mock=None) -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/v1")
    if ollama_mock is not None:
        app.state.ollama_client = ollama_mock
    return app


def test_api_generate_success():
    mock = AsyncMock()
    mock.chat = AsyncMock(return_value={"message": {"content": "Summary result."}})
    client = TestClient(_make_app(mock))

    resp = client.post("/v1/word/generate", json={
        "prompt": "Some long text.",
        "mode": "summarize",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "Summary result."
    assert data["mode"] == "summarize"


def test_api_generate_invalid_mode():
    mock = AsyncMock()
    client = TestClient(_make_app(mock))

    resp = client.post("/v1/word/generate", json={
        "prompt": "Text.",
        "mode": "dance",
    })
    assert resp.status_code == 400


def test_api_generate_no_ollama():
    app = FastAPI()
    app.include_router(router, prefix="/v1")
    # Don't set ollama_client on app.state
    client = TestClient(app)

    resp = client.post("/v1/word/generate", json={
        "prompt": "Text.",
        "mode": "summarize",
    })
    assert resp.status_code == 503


def test_api_generate_empty_prompt():
    mock = AsyncMock()
    client = TestClient(_make_app(mock))

    resp = client.post("/v1/word/generate", json={
        "prompt": "",
        "mode": "summarize",
    })
    assert resp.status_code == 422  # Pydantic validation
