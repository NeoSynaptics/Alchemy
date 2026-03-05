"""Async GUI-Actor client — talks to the GUI-Actor inference server on port 8200.

GUI-Actor (Microsoft, NeurIPS 2025) uses attention-based visual grounding.
It does NOT output text coordinates like other VLMs — instead it returns
normalized [0-1] click points from attention maps via a custom model class
(Qwen2VLForConditionalGenerationWithPointer).

The GUI-Actor server (alchemy/gui_actor/server.py) wraps this inference
behind a simple JSON API:
    POST /predict  →  {points: [[0.45, 0.32]], confidences: [0.92]}

This client handles the HTTP communication and coordinate scaling.

Available models on HuggingFace:
    - microsoft/GUI-Actor-7B-Qwen2.5-VL  (best, 44.6 on ScreenSpot-Pro)
    - microsoft/GUI-Actor-7B-Qwen2-VL
    - microsoft/GUI-Actor-3B-Qwen2.5-VL
    - microsoft/GUI-Actor-2B-Qwen2-VL
"""

from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass
class GUIActorPrediction:
    """Result from GUI-Actor inference."""
    x: int  # Pixel coordinate (scaled to screen)
    y: int  # Pixel coordinate (scaled to screen)
    norm_x: float  # Normalized [0-1]
    norm_y: float  # Normalized [0-1]
    confidence: float
    success: bool
    error: str | None = None


class GUIActorClient:
    """Async HTTP client for the GUI-Actor inference server."""

    def __init__(
        self,
        host: str = "http://localhost:8200",
        timeout: float = 30.0,
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
            raise RuntimeError("GUIActorClient not started — call start() first")
        return self._client

    async def predict(
        self,
        image: bytes,
        instruction: str,
        screen_width: int = 1280,
        screen_height: int = 720,
        topk: int = 1,
    ) -> GUIActorPrediction:
        """Send an image + instruction to GUI-Actor and get click coordinates.

        Args:
            image: Raw JPEG/PNG screenshot bytes.
            instruction: What to click (e.g., "Click the search button").
            screen_width: Viewport width for coordinate scaling.
            screen_height: Viewport height for coordinate scaling.
            topk: Number of candidate points to request (uses best one).

        Returns:
            GUIActorPrediction with pixel coordinates scaled to screen size.
        """
        client = self._ensure_client()

        b64_image = base64.b64encode(image).decode("ascii")
        payload = {
            "image": b64_image,
            "instruction": instruction,
            "topk": topk,
        }

        last_exc: Exception | None = None
        for attempt in range(self._retry_attempts):
            try:
                resp = await client.post("/predict", json=payload)
                resp.raise_for_status()
                data = resp.json()

                points = data.get("points", [])
                confidences = data.get("confidences", [])

                if not points:
                    return GUIActorPrediction(
                        x=0, y=0, norm_x=0.0, norm_y=0.0,
                        confidence=0.0, success=False,
                        error="No points returned by GUI-Actor",
                    )

                # Best point (highest confidence or first)
                norm_x, norm_y = points[0]
                conf = confidences[0] if confidences else 0.0

                # Scale normalized [0-1] to pixel coordinates
                px = round(norm_x * screen_width)
                py = round(norm_y * screen_height)
                px = min(max(px, 0), screen_width)
                py = min(max(py, 0), screen_height)

                return GUIActorPrediction(
                    x=px, y=py,
                    norm_x=norm_x, norm_y=norm_y,
                    confidence=conf, success=True,
                )

            except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                last_exc = e
                if attempt < self._retry_attempts - 1:
                    delay = self._retry_delay * (2 ** attempt)
                    logger.warning(
                        "GUI-Actor predict attempt %d/%d failed (%s), retrying in %.1fs",
                        attempt + 1, self._retry_attempts, e, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "GUI-Actor predict failed after %d attempts: %s",
                        self._retry_attempts, e,
                    )

        return GUIActorPrediction(
            x=0, y=0, norm_x=0.0, norm_y=0.0,
            confidence=0.0, success=False,
            error=f"All retries exhausted: {last_exc}",
        )

    async def ping(self) -> bool:
        """Check if GUI-Actor server is reachable."""
        try:
            client = self._ensure_client()
            resp = await client.get("/health", timeout=5.0)
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, RuntimeError):
            return False
