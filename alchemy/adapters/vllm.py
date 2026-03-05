"""Async vLLM client — OpenAI-compatible API for vision models like GUI-Actor 7B.

vLLM serves models on a local port with the standard /v1/chat/completions
endpoint. This client handles image encoding and the OpenAI message format
that vLLM expects (base64 images inside content arrays).

Usage:
    client = VLLMClient(host="http://localhost:8200")
    await client.start()
    result = await client.chat(
        model="gui-actor-7b",
        messages=[{"role": "user", "content": "Click the search button"}],
        images=[screenshot_bytes],
    )
"""

from __future__ import annotations

import asyncio
import base64
import logging

import httpx

logger = logging.getLogger(__name__)


class VLLMClient:
    """Async HTTP client for vLLM's OpenAI-compatible API."""

    def __init__(
        self,
        host: str = "http://localhost:8200",
        timeout: float = 60.0,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
    ):
        self._host = host.rstrip("/")
        self._timeout = timeout
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
            raise RuntimeError("VLLMClient not started — call start() first")
        return self._client

    async def chat(
        self,
        model: str,
        messages: list[dict],
        images: list[bytes] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 384,
    ) -> dict:
        """Send a chat completion request to vLLM.

        Args:
            model: Model name as served by vLLM.
            messages: Chat messages (OpenAI format).
            images: Raw image bytes to attach to the last user message.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.

        Returns:
            Dict with 'content' key containing the model's response text.
        """
        client = self._ensure_client()

        if images:
            messages = self._attach_images(messages, images)

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_exc: Exception | None = None
        for attempt in range(self._retry_attempts):
            try:
                resp = await client.post("/v1/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return {"message": {"content": content}}
            except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                last_exc = e
                if attempt < self._retry_attempts - 1:
                    delay = self._retry_delay * (2 ** attempt)
                    logger.warning(
                        "vLLM chat attempt %d/%d failed (%s), retrying in %.1fs",
                        attempt + 1, self._retry_attempts, e, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "vLLM chat failed after %d attempts: %s",
                        self._retry_attempts, e,
                    )

        raise last_exc  # type: ignore[misc]

    async def ping(self) -> bool:
        """Check if vLLM server is reachable."""
        try:
            client = self._ensure_client()
            resp = await client.get("/v1/models", timeout=5.0)
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, RuntimeError):
            return False

    async def list_models(self) -> list[dict]:
        """List models served by vLLM."""
        try:
            client = self._ensure_client()
            resp = await client.get("/v1/models")
            resp.raise_for_status()
            return resp.json().get("data", [])
        except Exception:
            return []

    @staticmethod
    def _attach_images(messages: list[dict], images: list[bytes]) -> list[dict]:
        """Attach base64 images to the last user message using OpenAI vision format.

        Converts: {"role": "user", "content": "text"}
        To:       {"role": "user", "content": [
                      {"type": "text", "text": "..."},
                      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
                  ]}
        """
        messages = [m.copy() for m in messages]
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                text = messages[i].get("content", "")
                content_parts: list[dict] = [{"type": "text", "text": text}]
                for img in images:
                    b64 = base64.b64encode(img).decode("ascii")
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    })
                messages[i]["content"] = content_parts
                break
        return messages
