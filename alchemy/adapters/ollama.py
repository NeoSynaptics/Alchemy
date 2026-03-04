"""Async Ollama client — talks to the local Ollama server for model inference.

Uses the native Ollama API (/api/chat, /api/tags, /api/pull) for full control
over keep_alive, streaming, and model management. Images are base64-encoded
internally — callers pass raw PNG/JPEG bytes.
"""

from __future__ import annotations

import asyncio
import base64
import json
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
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
    ):
        self._host = host.rstrip("/")
        self._timeout = timeout
        self._keep_alive = keep_alive
        self._retry_attempts = retry_attempts
        self._retry_delay = retry_delay
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
        options: dict | None = None,
    ) -> dict:
        """Send a chat completion request with retry logic.

        Args:
            model: Ollama model name.
            messages: Chat messages in Ollama format.
            images: Optional list of raw image bytes to attach to the last user message.
            options: Optional Ollama options (temperature, num_predict, etc).

        Returns:
            Full Ollama response dict with 'message', 'total_duration', etc.

        Raises:
            After exhausting retries: the last exception encountered.
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
        if options:
            payload["options"] = options

        last_exc: Exception | None = None
        for attempt in range(self._retry_attempts):
            try:
                resp = await client.post("/api/chat", json=payload)
                resp.raise_for_status()
                return resp.json()
            except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                last_exc = e
                if attempt < self._retry_attempts - 1:
                    delay = self._retry_delay * (2 ** attempt)
                    logger.warning(
                        "Ollama chat attempt %d/%d failed (%s), retrying in %.1fs",
                        attempt + 1, self._retry_attempts, e, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error("Ollama chat failed after %d attempts: %s", self._retry_attempts, e)

        raise last_exc  # type: ignore[misc]

    async def chat_think(
        self,
        model: str,
        messages: list[dict],
        think: bool = True,
        options: dict | None = None,
        seed: int | None = None,
    ) -> dict:
        """Chat with Qwen3 think mode support and retry logic.

        Args:
            model: Ollama model name (e.g., "qwen3:14b").
            messages: Chat messages in Ollama format.
            think: Enable thinking mode (Qwen3 puts reasoning in 'thinking' field).
            options: Optional Ollama options (temperature, num_predict, etc).
            seed: Optional seed for reproducible output.

        Returns:
            Dict with 'content', 'thinking', and 'total_duration' keys.
        """
        client = self._ensure_client()
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": self._keep_alive,
            "think": think,
        }
        if options:
            opts = dict(options)
            if seed is not None:
                opts["seed"] = seed
            payload["options"] = opts
        elif seed is not None:
            payload["options"] = {"seed": seed}

        last_exc: Exception | None = None
        for attempt in range(self._retry_attempts):
            try:
                resp = await client.post("/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
                msg = data.get("message", {})
                return {
                    "content": msg.get("content", ""),
                    "thinking": msg.get("thinking", ""),
                    "total_duration": data.get("total_duration"),
                }
            except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                last_exc = e
                if attempt < self._retry_attempts - 1:
                    delay = self._retry_delay * (2 ** attempt)
                    logger.warning(
                        "Ollama chat_think attempt %d/%d failed (%s), retrying in %.1fs",
                        attempt + 1, self._retry_attempts, e, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "Ollama chat_think failed after %d attempts: %s",
                        self._retry_attempts, e,
                    )

        raise last_exc  # type: ignore[misc]

    async def chat_stream(
        self,
        model: str,
        messages: list[dict],
        images: list[bytes] | None = None,
        options: dict | None = None,
        stop_at: str | None = None,
    ) -> str:
        """Stream chat completion, optionally stopping early when stop_at is found.

        Returns the full accumulated response text. If stop_at is provided,
        stops reading as soon as that string appears in the accumulated output —
        saves time by not waiting for the model to finish generating.
        """
        client = self._ensure_client()

        if images:
            messages = self._attach_images(messages, images)

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "keep_alive": self._keep_alive,
        }
        if options:
            payload["options"] = options

        accumulated = ""
        last_exc: Exception | None = None

        for attempt in range(self._retry_attempts):
            try:
                accumulated = ""
                async with client.stream("POST", "/api/chat", json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        accumulated += token

                        if chunk.get("done"):
                            break

                        # Early stop: once we have a complete Action line, stop reading
                        if stop_at and stop_at in accumulated:
                            # Check if we have a complete action (ends with closing paren)
                            after_action = accumulated[accumulated.index(stop_at):]
                            if ")" in after_action:
                                break
                return accumulated
            except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                last_exc = e
                if attempt < self._retry_attempts - 1:
                    delay = self._retry_delay * (2 ** attempt)
                    logger.warning(
                        "Ollama stream attempt %d/%d failed (%s), retrying in %.1fs",
                        attempt + 1, self._retry_attempts, e, delay,
                    )
                    await asyncio.sleep(delay)

        raise last_exc  # type: ignore[misc]

    async def chat_stream_raw(
        self,
        model: str,
        messages: list[dict],
        images: list[bytes] | None = None,
        options: dict | None = None,
    ) -> AsyncGenerator[dict, None]:
        """Yield raw streaming chunks for low-level consumers."""
        client = self._ensure_client()

        if images:
            messages = self._attach_images(messages, images)

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "keep_alive": self._keep_alive,
        }
        if options:
            payload["options"] = options

        async with client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.strip():
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
