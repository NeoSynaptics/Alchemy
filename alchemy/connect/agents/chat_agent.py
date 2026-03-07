"""ChatAgent — local LLM chat through the AlchemyConnect tunnel.

Routes text messages to the voice SmartRouter (which picks the best
local model). Streams responses back token-by-token.

Protocol:
    -> {agent: "chat", type: "message", payload: {text: "..."}}
    <- {agent: "chat", type: "token",   payload: {text: "partial..."}}
    <- {agent: "chat", type: "done",    payload: {text: "full response", model: "qwen3:14b"}}
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from alchemy.connect.protocol import AlchemyMessage
from alchemy.connect.router import ConnectAgent

logger = logging.getLogger(__name__)


class ChatAgent(ConnectAgent):
    """Chat with local LLMs through the tunnel."""

    def __init__(self, app_state: Any) -> None:
        self._app_state = app_state

    @property
    def agent_id(self) -> str:
        return "chat"

    def describe(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "description": "Chat with local LLM (Qwen3 14B or routed model)",
            "types": {
                "message": "Send a text message, get streamed response",
            },
        }

    async def handle(
        self,
        msg: AlchemyMessage,
        device_id: str,
    ) -> AsyncIterator[AlchemyMessage]:
        if msg.type != "message":
            yield AlchemyMessage(
                agent="chat",
                type="error",
                payload={"reason": f"Unknown type: {msg.type}"},
                ref=msg.id,
            )
            return

        text = msg.payload.get("text", "").strip()
        if not text:
            yield AlchemyMessage(
                agent="chat",
                type="error",
                payload={"reason": "Empty message"},
                ref=msg.id,
            )
            return

        # Get the SmartRouter from voice system
        voice_system = getattr(self._app_state, "voice_system", None)
        smart_router = None
        if voice_system and hasattr(voice_system, "_router"):
            smart_router = voice_system._router

        if not smart_router:
            # Fallback: try Ollama directly
            ollama = getattr(self._app_state, "ollama_client", None)
            if not ollama:
                yield AlchemyMessage(
                    agent="chat",
                    type="error",
                    payload={"reason": "No LLM available"},
                    ref=msg.id,
                )
                return

            # Direct Ollama chat (no SmartRouter)
            full_text = ""
            try:
                async for chunk in ollama.chat_stream(
                    model="qwen3:14b",
                    messages=[{"role": "user", "content": text}],
                    options={"temperature": 0.7, "num_predict": 1024, "think": False},
                ):
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        full_text += token
                        yield AlchemyMessage(
                            agent="chat",
                            type="token",
                            payload={"text": token},
                            ref=msg.id,
                        )

                yield AlchemyMessage(
                    agent="chat",
                    type="done",
                    payload={"text": full_text, "model": "qwen3:14b"},
                    ref=msg.id,
                )
            except Exception as e:
                logger.exception("ChatAgent Ollama error")
                yield AlchemyMessage(
                    agent="chat",
                    type="error",
                    payload={"reason": str(e)},
                    ref=msg.id,
                )
            return

        # Use SmartRouter (preferred path)
        from alchemy.voice.models.schemas import ChatRequest

        request = ChatRequest(message=text)
        full_text = ""
        model_used = ""

        try:
            async for chunk in smart_router.route_stream(request):
                token = chunk.content or ""
                if token:
                    full_text += token
                    yield AlchemyMessage(
                        agent="chat",
                        type="token",
                        payload={"text": token},
                        ref=msg.id,
                    )
                if chunk.model:
                    model_used = chunk.model

            yield AlchemyMessage(
                agent="chat",
                type="done",
                payload={"text": full_text, "model": model_used},
                ref=msg.id,
            )
        except Exception as e:
            logger.exception("ChatAgent SmartRouter error")
            yield AlchemyMessage(
                agent="chat",
                type="error",
                payload={"reason": str(e)},
                ref=msg.id,
            )
