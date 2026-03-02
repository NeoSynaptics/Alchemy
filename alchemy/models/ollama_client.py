"""Async Ollama client — talks to the local Ollama server for model inference.

Uses the native Ollama API (/api/chat, /api/tags, /api/pull) for full control
over keep_alive, streaming, and model management. Images are base64-encoded
internally — callers pass raw PNG bytes.
"""

from __future__ import annotations

import base64
import logging
from typing import AsyncGenerator

import httpx

logger = logging.getLogger(__name__)


class OllamaClient:
    """Async HTTP client for Ollama's native API."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        timeout: float = 120.0,
        keep_alive: str = "10m",
    ):
        self._host = host.rstrip("/")
        self._timeout = timeout
        self._keep_alive = keep_alive
        self._client: httpx.AsyncClient | None = None

    async def start(self):
        """Create the HTTP connection pool."""
        self._client = httpx.AsyncClient(
            base_url=self._host,
            timeout=httpx.Timeout(self._timeout, connect=10.0),
        )

    async def close(self):
        """Close the HTTP connection pool."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("OllamaClient not started — call start() first")
        return self._client

    async def chat(
        self,
        model: str,
        messages: list[dict],
        images: list[bytes] | None = None,
    ) -> dict:
        """Send a chat completion request and return the full response.

        Args:
            model: Ollama model name (e.g., "avil/UI-TARS").
            messages: Chat messages in Ollama format.
            images: Optional list of raw PNG bytes to attach to the last user message.

        Returns:
            Full Ollama response dict with 'message', 'total_duration', etc.
        """
        client = self._ensure_client()

        if images:
            messages = self._attach_images(messages, images)

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": self._keep_alive,
        }

        resp = await client.post("/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def chat_stream(
        self,
        model: str,
        messages: list[dict],
        images: list[bytes] | None = None,
    ) -> AsyncGenerator[dict, None]:
        """Stream chat completion responses as they arrive."""
        client = self._ensure_client()

        if images:
            messages = self._attach_images(messages, images)

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "keep_alive": self._keep_alive,
        }

        async with client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.strip():
                    import json
                    yield json.loads(line)

    async def list_models(self) -> list[dict]:
        """List locally available models."""
        client = self._ensure_client()
        resp = await client.get("/api/tags")
        resp.raise_for_status()
        return resp.json().get("models", [])

    async def is_model_available(self, model: str) -> bool:
        """Check if a specific model is available locally."""
        models = await self.list_models()
        return any(m.get("name", "").startswith(model) for m in models)

    async def show_model(self, model: str) -> dict:
        """Get model metadata (size, quantization, etc)."""
        client = self._ensure_client()
        resp = await client.post("/api/show", json={"name": model})
        resp.raise_for_status()
        return resp.json()

    async def pull_model(self, model: str) -> AsyncGenerator[dict, None]:
        """Pull/download a model. Yields progress dicts."""
        client = self._ensure_client()
        async with client.stream(
            "POST", "/api/pull", json={"name": model, "stream": True}
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.strip():
                    import json
                    yield json.loads(line)

    async def ping(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            client = self._ensure_client()
            resp = await client.get("/", timeout=5.0)
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, RuntimeError):
            return False

    @staticmethod
    def _attach_images(messages: list[dict], images: list[bytes]) -> list[dict]:
        """Attach base64-encoded images to the last user message."""
        messages = [m.copy() for m in messages]
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                messages[i]["images"] = [
                    base64.b64encode(img).decode("ascii") for img in images
                ]
                break
        return messages
