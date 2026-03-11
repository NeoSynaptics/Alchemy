"""Chat endpoints — /chat (non-streaming), /chat/stream (SSE), /chat/reaction (RLHF)."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from alchemy.voice.models.schemas import ChatRequest, ChatResponse
from alchemy.voice.reactions import VoiceReaction, get_reaction_logger
from alchemy.voice.router.router import SmartRouter

router = APIRouter(prefix="/chat", tags=["chat"])


def _get_router(req: Request) -> SmartRouter:
    """Get the voice SmartRouter from app state."""
    voice_system = getattr(req.app.state, "voice_system", None)
    if voice_system and voice_system._router:
        return voice_system._router
    # Fallback: check direct app.state.router (legacy)
    return getattr(req.app.state, "router", None)


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request) -> ChatResponse:
    """Non-streaming chat. Returns full response after inference completes."""
    smart_router = _get_router(req)
    if not smart_router:
        return ChatResponse(content="Voice system not available", model="none")
    return await smart_router.route(request)


@router.post("/stream")
async def chat_stream(request: ChatRequest, req: Request) -> StreamingResponse:
    """Streaming chat via Server-Sent Events. Primary endpoint for real-time UX."""
    smart_router = _get_router(req)

    async def event_generator():
        if not smart_router:
            yield 'data: {"error": "Voice system not available"}\n\n'
            return
        async for chunk in smart_router.route_stream(request):
            data = chunk.model_dump_json()
            yield f"data: {data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/reaction")
async def submit_reaction(reaction: VoiceReaction):
    """Log a user reaction to a voice response for RLHF preference data."""
    reaction_logger = get_reaction_logger()
    if not reaction_logger:
        return {"status": "skipped", "reason": "reaction logging not initialized"}
    await reaction_logger.log_reaction(reaction)
    return {"status": "logged", "sentiment": reaction.sentiment.value}
