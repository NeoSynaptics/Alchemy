"""Smart router — routes user requests to the optimal model.

Flow:
1. Keyword pre-filter (zero cost, <1ms)
2. If obvious → route directly (skip 14B)
3. Otherwise → 14B with tag instruction → parse tag from first tokens
4. Stream conversation / reroute GUI tasks / handle system commands
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncGenerator

from alchemy.voice.models.conversation import ConversationManager
from alchemy.voice.models.provider import ModelProvider
from alchemy.voice.models.registry import ModelRegistry
from alchemy.voice.models.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ModelCapability,
    ModelLocation,
    RouteDecision,
    RouteIntent,
    StreamChunk,
)
from alchemy.voice.router.cascade import CascadeStrategy
from alchemy.voice.router.classifier import (
    CLASSIFIER_INSTRUCTION,
    classify_from_keywords,
    parse_intent_tag,
)

logger = logging.getLogger(__name__)


class SmartRouter:
    """Routes user requests to the optimal model based on intent and capability."""

    def __init__(
        self,
        registry: ModelRegistry,
        providers: dict[ModelLocation, ModelProvider],
        conversation_manager: ConversationManager,
        cascades: list[CascadeStrategy] | None = None,
        knowledge_retriever=None,
        event_reporter=None,
    ) -> None:
        self._registry = registry
        self._providers = providers
        self._conversations = conversation_manager
        self._cascades = cascades or []
        self._retriever = knowledge_retriever
        self._reporter = event_reporter

    @property
    def registry(self) -> ModelRegistry:
        return self._registry

    async def route(self, request: ChatRequest) -> ChatResponse:
        """Non-streaming route: classify, infer, return full response."""
        decision = self._classify(request)

        if decision.intent == RouteIntent.GUI_TASK:
            return await self._handle_gui_task(request, decision)

        if decision.intent == RouteIntent.SYSTEM_COMMAND:
            return self._handle_system_command(request, decision)

        # CONVERSATION or UNCLEAR → 14B
        return await self._handle_conversation(request, decision)

    async def route_stream(
        self, request: ChatRequest
    ) -> AsyncGenerator[StreamChunk, None]:
        """Streaming route. Primary endpoint for real-time UX."""
        keyword_intent = classify_from_keywords(request.message)

        if keyword_intent == RouteIntent.GUI_TASK:
            decision = RouteDecision(
                intent=RouteIntent.GUI_TASK,
                target_model="ui-tars:72b",
                target_location=ModelLocation.CPU_REMOTE,
                confidence=0.95,
                reasoning="Keyword match: GUI task",
                should_stream=False,
            )
            response = await self._handle_gui_task(request, decision)
            yield StreamChunk(
                content=response.message,
                done=True,
                model_used=response.model_used,
                inference_ms=response.inference_ms,
            )
            return

        if keyword_intent == RouteIntent.SYSTEM_COMMAND:
            decision = RouteDecision(
                intent=RouteIntent.SYSTEM_COMMAND,
                target_model="internal",
                target_location=ModelLocation.GPU_LOCAL,
                confidence=0.95,
                reasoning="Keyword match: system command",
                should_stream=False,
            )
            response = self._handle_system_command(request, decision)
            yield StreamChunk(content=response.message, done=True, model_used="internal")
            return

        # Ambiguous → let 14B classify via tag prefix
        conv_model = self._registry.get_default(ModelCapability.CONVERSATION)
        if not conv_model:
            yield StreamChunk(content="No conversational model available.", done=True)
            return

        provider = self._providers.get(conv_model.location)
        if not provider:
            yield StreamChunk(content="Model provider unavailable.", done=True)
            return

        # Retrieve relevant knowledge from NEO-RX (non-blocking, graceful)
        knowledge_context = []
        if self._retriever:
            knowledge_context = await self._retriever.retrieve(request.message)

        self._conversations.add_user_message(request.conversation_id, request.message)
        messages = self._conversations.get_messages(
            request.conversation_id,
            knowledge_context=knowledge_context,
        )

        # Inject classifier instruction into system prompt
        messages[0] = ChatMessage(
            role="system",
            content=messages[0].content + CLASSIFIER_INSTRUCTION,
        )

        full_response = ""
        intent_parsed = False
        t0 = time.monotonic()

        async for chunk_text in provider.generate_stream(
            model=conv_model.name,
            messages=messages,
            endpoint=conv_model.endpoint,
            think=False,
        ):
            full_response += chunk_text

            if not intent_parsed and len(full_response) > 5:
                intent, cleaned = parse_intent_tag(full_response)
                if intent != RouteIntent.UNCLEAR:
                    intent_parsed = True
                    full_response = cleaned

                    if intent == RouteIntent.GUI_TASK:
                        decision = RouteDecision(
                            intent=RouteIntent.GUI_TASK,
                            target_model="ui-tars:72b",
                            target_location=ModelLocation.CPU_REMOTE,
                            confidence=0.85,
                            reasoning="14B classified as GUI task",
                            should_stream=False,
                        )
                        gui_response = await self._handle_gui_task(request, decision)
                        if cleaned.strip():
                            yield StreamChunk(content=cleaned.strip() + "\n\n")
                        yield StreamChunk(
                            content=gui_response.message,
                            done=True,
                            model_used=gui_response.model_used,
                            inference_ms=gui_response.inference_ms,
                        )
                        self._conversations.add_assistant_message(
                            request.conversation_id,
                            (cleaned.strip() + "\n\n" + gui_response.message).strip(),
                        )
                        return

                    if cleaned.strip():
                        yield StreamChunk(content=cleaned)
                    continue

            if intent_parsed:
                yield StreamChunk(content=chunk_text)
            elif len(full_response) > 50:
                # No tag after 50 chars → assume conversation
                intent_parsed = True
                yield StreamChunk(content=full_response)

        elapsed_ms = (time.monotonic() - t0) * 1000

        # Check cascades for escalation
        decision = RouteDecision(
            intent=RouteIntent.CONVERSATION,
            target_model=conv_model.name,
            target_location=conv_model.location,
            confidence=0.8,
            reasoning="14B conversation",
            should_stream=True,
            escalation_possible=True,
        )
        for cascade in self._cascades:
            if cascade.should_escalate(decision, full_response):
                escalation = cascade.escalation_target(decision)
                gui_response = await self._handle_gui_task(request, escalation)
                yield StreamChunk(
                    content=f"\n\n{gui_response.message}",
                    done=True,
                    model_used=gui_response.model_used,
                    inference_ms=gui_response.inference_ms,
                )
                self._conversations.add_assistant_message(
                    request.conversation_id,
                    full_response + "\n\n" + gui_response.message,
                )
                return

        self._conversations.add_assistant_message(
            request.conversation_id, full_response
        )

        # Report conversation to NEO-RX timeline (fire-and-forget)
        if self._reporter:
            asyncio.create_task(
                self._reporter.report_conversation(
                    request.conversation_id, "user", request.message
                )
            )
            asyncio.create_task(
                self._reporter.report_conversation(
                    request.conversation_id, "assistant", full_response
                )
            )

        yield StreamChunk(
            content="", done=True, model_used=conv_model.name, inference_ms=elapsed_ms
        )

    def _classify(self, request: ChatRequest) -> RouteDecision:
        """Keyword-based classification for non-streaming route."""
        keyword_intent = classify_from_keywords(request.message)

        if keyword_intent == RouteIntent.GUI_TASK:
            return RouteDecision(
                intent=RouteIntent.GUI_TASK,
                target_model="ui-tars:72b",
                target_location=ModelLocation.CPU_REMOTE,
                confidence=0.95,
                reasoning="Keyword match: GUI task",
                should_stream=False,
            )

        if keyword_intent == RouteIntent.SYSTEM_COMMAND:
            return RouteDecision(
                intent=RouteIntent.SYSTEM_COMMAND,
                target_model="internal",
                target_location=ModelLocation.GPU_LOCAL,
                confidence=0.95,
                reasoning="Keyword match: system command",
                should_stream=False,
            )

        conv_model = self._registry.get_default(ModelCapability.CONVERSATION)
        model_name = conv_model.name if conv_model else "unknown"
        return RouteDecision(
            intent=RouteIntent.CONVERSATION,
            target_model=model_name,
            target_location=ModelLocation.GPU_LOCAL,
            confidence=0.7,
            reasoning="Default to 14B with inline classification",
            should_stream=True,
            escalation_possible=True,
        )

    async def _handle_conversation(
        self, request: ChatRequest, decision: RouteDecision
    ) -> ChatResponse:
        """Full non-streaming conversation via 14B."""
        conv_model = self._registry.get_default(ModelCapability.CONVERSATION)
        if not conv_model:
            return ChatResponse(
                message="No conversational model available.",
                conversation_id=request.conversation_id,
                model_used="none",
                route_decision=decision,
            )

        provider = self._providers.get(conv_model.location)
        if not provider:
            return ChatResponse(
                message="Model provider unavailable.",
                conversation_id=request.conversation_id,
                model_used="none",
                route_decision=decision,
            )

        # Retrieve relevant knowledge from NEO-RX
        knowledge_context = []
        if self._retriever:
            knowledge_context = await self._retriever.retrieve(request.message)

        self._conversations.add_user_message(request.conversation_id, request.message)
        messages = self._conversations.get_messages(
            request.conversation_id,
            knowledge_context=knowledge_context,
        )
        messages[0] = ChatMessage(
            role="system",
            content=messages[0].content + CLASSIFIER_INSTRUCTION,
        )

        raw_text, elapsed_ms = await provider.generate(
            model=conv_model.name,
            messages=messages,
            endpoint=conv_model.endpoint,
            think=False,
        )

        _, cleaned = parse_intent_tag(raw_text)
        self._conversations.add_assistant_message(request.conversation_id, cleaned)

        # Report to NEO-RX timeline
        if self._reporter:
            asyncio.create_task(
                self._reporter.report_conversation(
                    request.conversation_id, "user", request.message
                )
            )
            asyncio.create_task(
                self._reporter.report_conversation(
                    request.conversation_id, "assistant", cleaned
                )
            )

        return ChatResponse(
            message=cleaned,
            conversation_id=request.conversation_id,
            model_used=conv_model.name,
            route_decision=decision,
            inference_ms=elapsed_ms,
        )

    async def _handle_gui_task(
        self, request: ChatRequest, decision: RouteDecision
    ) -> ChatResponse:
        """Submit a GUI task to Alchemy."""
        provider = self._providers.get(ModelLocation.CPU_REMOTE)
        if not provider:
            return ChatResponse(
                message="I'd like to help with that, but the background agent (Alchemy) isn't available right now.",
                conversation_id=request.conversation_id,
                model_used="internal",
                route_decision=decision,
            )

        messages = [ChatMessage(role="user", content=request.message)]
        try:
            text, elapsed_ms = await provider.generate("ui-tars:72b", messages)
        except Exception:
            logger.warning("Alchemy unreachable for GUI task: %s", request.message)
            return ChatResponse(
                message="I'd like to help with that, but the background agent (Alchemy) isn't reachable right now. Is it running?",
                conversation_id=request.conversation_id,
                model_used="internal",
                route_decision=decision,
            )

        self._conversations.add_user_message(request.conversation_id, request.message)
        self._conversations.add_assistant_message(request.conversation_id, text)

        return ChatResponse(
            message=text,
            conversation_id=request.conversation_id,
            model_used="ui-tars:72b",
            route_decision=decision,
            inference_ms=elapsed_ms,
        )

    @staticmethod
    def _handle_system_command(
        request: ChatRequest, decision: RouteDecision
    ) -> ChatResponse:
        """Handle system commands (stub — Phase 2+)."""
        return ChatResponse(
            message=f"System command received: '{request.message}'. (Not yet implemented.)",
            conversation_id=request.conversation_id,
            model_used="internal",
            route_decision=decision,
        )
