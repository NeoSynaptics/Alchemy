"""Model providers — abstract interface + Ollama and Alchemy implementations.

Each provider handles one type of model location. The OllamaProvider supports
per-model endpoint overrides for dual-host setups (Windows Ollama + WSL2 Ollama).
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import AsyncGenerator

import httpx

from alchemy.voice.models.schemas import ChatMessage

logger = logging.getLogger(__name__)


class ModelProvider(ABC):
    """Abstract interface for running inference against a model."""

    @abstractmethod
    async def start(self) -> None:
        """Initialize connections."""

    @abstractmethod
    async def close(self) -> None:
        """Clean up connections."""

    @abstractmethod
    async def generate(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        endpoint: str | None = None,
        think: bool | None = None,
    ) -> tuple[str, float]:
        """Non-streaming generation. Returns (response_text, inference_ms)."""

    @abstractmethod
    async def generate_stream(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        endpoint: str | None = None,
        think: bool | None = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming generation. Yields text chunks."""

    @abstractmethod
    async def is_available(self, endpoint: str | None = None) -> bool:
        """Check if this provider is reachable."""


class OllamaProvider(ModelProvider):
    """Provider for GPU models via Ollama /api/chat.

    Supports per-request endpoint override so different models can talk to
    different Ollama instances (e.g., Windows GPU vs WSL2 CPU).
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        timeout: float = 120.0,
        keep_alive: str = "30m",
    ) -> None:
        self._default_host = host.rstrip("/")
        self._timeout = timeout
        self._keep_alive = keep_alive
        self._clients: dict[str, httpx.AsyncClient] = {}

    async def start(self) -> None:
        self._clients[self._default_host] = httpx.AsyncClient(
            base_url=self._default_host,
            timeout=httpx.Timeout(self._timeout, connect=10.0),
        )

    async def close(self) -> None:
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()

    def _get_client(self, endpoint: str | None) -> httpx.AsyncClient:
        host = (endpoint or self._default_host).rstrip("/")
        if host not in self._clients:
            # Lazy-create client for a new endpoint (dual-host support)
            self._clients[host] = httpx.AsyncClient(
                base_url=host,
                timeout=httpx.Timeout(self._timeout, connect=10.0),
            )
        return self._clients[host]

    @staticmethod
    def _to_ollama_messages(messages: list[ChatMessage]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def _build_payload(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        stream: bool,
        temperature: float = 0.7,
        think: bool | None = None,
    ) -> dict:
        """Build the Ollama /api/chat request payload."""
        payload: dict = {
            "model": model,
            "messages": self._to_ollama_messages(messages),
            "stream": stream,
            "keep_alive": self._keep_alive,
            "options": {"temperature": temperature},
        }
        if think is not None:
            payload["think"] = think
        return payload

    async def generate(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        endpoint: str | None = None,
        think: bool | None = None,
    ) -> tuple[str, float]:
        client = self._get_client(endpoint)
        payload = self._build_payload(
            model, messages, stream=False, temperature=temperature, think=think,
        )
        t0 = time.monotonic()
        resp = await client.post("/api/chat", json=payload)
        resp.raise_for_status()
        elapsed_ms = (time.monotonic() - t0) * 1000
        data = resp.json()
        return data["message"]["content"], elapsed_ms

    async def generate_stream(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        endpoint: str | None = None,
        think: bool | None = None,
    ) -> AsyncGenerator[str, None]:
        client = self._get_client(endpoint)
        payload = self._build_payload(
            model, messages, stream=True, temperature=temperature, think=think,
        )
        async with client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.strip():
                    chunk = json.loads(line)
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content

    async def is_available(self, endpoint: str | None = None) -> bool:
        try:
            client = self._get_client(endpoint)
            resp = await client.get("/", timeout=5.0)
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False


class AlchemyProvider(ModelProvider):
    """Provider for CPU models via Alchemy's API.

    Wraps AlchemyClient for vision tasks. The 'generate' methods translate
    a goal into a vision task submission.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url
        self._timeout = timeout
        self._alchemy_client: object | None = None

    async def start(self) -> None:
        from alchemy.voice.bridge.alchemy_client import AlchemyClient

        self._alchemy_client = AlchemyClient(
            base_url=self._base_url, timeout=self._timeout
        )

    async def close(self) -> None:
        if self._alchemy_client is not None:
            await self._alchemy_client.close()
            self._alchemy_client = None

    async def generate(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        endpoint: str | None = None,
        think: bool | None = None,
    ) -> tuple[str, float]:
        if not self._alchemy_client:
            raise RuntimeError("AlchemyProvider not started")
        goal = messages[-1].content if messages else ""
        t0 = time.monotonic()
        result = await self._alchemy_client.submit_task(goal=goal)
        elapsed_ms = (time.monotonic() - t0) * 1000
        return (
            f"Task submitted to Alchemy (ID: {result.task_id}). "
            f"Status: {result.status.value}. I'll notify you when it's done.",
            elapsed_ms,
        )

    async def generate_stream(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        endpoint: str | None = None,
        think: bool | None = None,
    ) -> AsyncGenerator[str, None]:
        text, _ = await self.generate(model, messages, temperature=temperature)
        yield text

    async def is_available(self, endpoint: str | None = None) -> bool:
        if not self._alchemy_client:
            return False
        try:
            await self._alchemy_client.models()
            return True
        except Exception:
            return False


class GatewayProvider(ModelProvider):
    """Routes Voice inference through the APU inference gateway.

    Wraps a _CallerProxy (from gateway.with_caller("voice", priority=8))
    and adapts it to the ModelProvider interface that SmartRouter expects.
    Gives Voice: model loading, VRAM management, per-caller metrics — zero
    changes to SmartRouter or any voice module.
    """

    def __init__(self, proxy: object) -> None:
        self._proxy = proxy

    async def start(self) -> None:
        pass  # Gateway manages its own lifecycle

    async def close(self) -> None:
        pass  # Gateway manages its own lifecycle

    @staticmethod
    def _to_dicts(messages: list[ChatMessage]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    async def generate(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        endpoint: str | None = None,
        think: bool | None = None,
    ) -> tuple[str, float]:
        options: dict = {"temperature": temperature}
        kwargs: dict = {"options": options}
        if think is not None:
            kwargs["think"] = think
        t0 = time.monotonic()
        result = await self._proxy.chat(model, self._to_dicts(messages), **kwargs)
        elapsed_ms = (time.monotonic() - t0) * 1000
        return result.get("message", {}).get("content", ""), elapsed_ms

    async def generate_stream(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        endpoint: str | None = None,
        think: bool | None = None,
    ) -> AsyncGenerator[str, None]:
        options: dict = {"temperature": temperature}
        kwargs: dict = {"options": options}
        if think is not None:
            kwargs["think"] = think
        async for chunk in self._proxy.chat_stream_raw(
            model, self._to_dicts(messages), **kwargs,
        ):
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content

    async def is_available(self, endpoint: str | None = None) -> bool:
        return True  # Gateway is available if the server started
