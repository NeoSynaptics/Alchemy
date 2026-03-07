"""ImageAgent — AI image generation through the AlchemyConnect tunnel.

STUB: Flux is not installed yet. This agent reports status and validates
the protocol. When Flux is wired, it will:
  1. Acquire GPU guard (semaphore)
  2. Load model via APU
  3. Run diffusers pipeline
  4. Base64 encode result
  5. Stream progress + final image back to phone

Protocol:
    -> {agent: "image", type: "generate", payload: {prompt: "...", width: 512, height: 512}}
    <- {agent: "image", type: "progress", payload: {step: 5, total: 20}}
    <- {agent: "image", type: "result",   payload: {image_b64: "...", width: 512, height: 512, model: "flux-dev"}}

    -> {agent: "image", type: "status"}
    <- {agent: "image", type: "status", payload: {available: false, reason: "..."}}
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from alchemy.connect.protocol import AlchemyMessage
from alchemy.connect.router import ConnectAgent

logger = logging.getLogger(__name__)


class ImageAgent(ConnectAgent):
    """AI image generation via local GPU (Flux/SD)."""

    def __init__(self, app_state: Any, gpu_guard: Any = None) -> None:
        self._app_state = app_state
        self._gpu_guard = gpu_guard  # asyncio.Semaphore from hub

    @property
    def agent_id(self) -> str:
        return "image"

    def describe(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "description": "AI image generation (Flux/SD on local GPU)",
            "available": self._is_available(),
            "types": {
                "generate": "Generate image from text prompt",
                "status": "Check if image generation is available",
            },
        }

    def _is_available(self) -> bool:
        """Check if an image generation model is loaded and ready."""
        # Future: check APU for loaded Flux/SD model
        return False

    async def handle(
        self,
        msg: AlchemyMessage,
        device_id: str,
    ) -> AsyncIterator[AlchemyMessage]:
        if msg.type == "status":
            yield AlchemyMessage(
                agent="image",
                type="status",
                payload={
                    "available": self._is_available(),
                    "reason": "Image generation model not installed yet (Flux coming soon)",
                    "supported_models": ["flux-dev", "flux-schnell", "sdxl"],
                },
                ref=msg.id,
            )
            return

        if msg.type == "generate":
            if not self._is_available():
                yield AlchemyMessage(
                    agent="image",
                    type="error",
                    payload={
                        "reason": "Image generation not available — no model loaded",
                        "hint": "Install Flux and configure in APU fleet",
                    },
                    ref=msg.id,
                )
                return

            # Future implementation:
            # prompt = msg.payload.get("prompt", "")
            # width = msg.payload.get("width", 512)
            # height = msg.payload.get("height", 512)
            # steps = msg.payload.get("steps", 20)
            #
            # async with self._gpu_guard:
            #     # Load model via APU if needed
            #     # Run diffusers pipeline with progress callback
            #     for step in range(steps):
            #         yield AlchemyMessage(agent="image", type="progress",
            #             payload={"step": step + 1, "total": steps}, ref=msg.id)
            #     # Encode result
            #     image_b64 = base64.b64encode(jpeg_bytes).decode()
            #     yield AlchemyMessage(agent="image", type="result",
            #         payload={"image_b64": image_b64, "width": width,
            #                  "height": height, "model": "flux-dev"}, ref=msg.id)
            return

        yield AlchemyMessage(
            agent="image",
            type="error",
            payload={"reason": f"Unknown type: {msg.type}"},
            ref=msg.id,
        )
