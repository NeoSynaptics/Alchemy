"""Voice control endpoints — start/stop/status for the voice pipeline."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(prefix="/voice", tags=["voice"])


class VoiceStatusResponse(BaseModel):
    running: bool
    mode: str
    pipeline_state: str
    tts_engine: str
    wake_word: str
    conversation_id: str | None = None


class VoiceModeRequest(BaseModel):
    mode: str  # "conversation", "command", "dictation", "muted"


@router.get("/status", response_model=VoiceStatusResponse)
async def voice_status(req: Request) -> VoiceStatusResponse:
    """Get current voice system status."""
    voice_system = getattr(req.app.state, "voice_system", None)
    if not voice_system:
        return VoiceStatusResponse(
            running=False, mode="muted", pipeline_state="idle",
            tts_engine="none", wake_word="",
        )
    status = voice_system.status()
    return VoiceStatusResponse(**status.to_dict())


@router.post("/start")
async def voice_start(req: Request):
    """Start the voice pipeline."""
    voice_system = getattr(req.app.state, "voice_system", None)
    if not voice_system:
        return {"error": "Voice not available (voice_enabled=false or no audio device)"}

    if voice_system.is_running:
        return {"status": "already_running", "mode": voice_system.mode.value}

    await voice_system.start()
    return {"status": "started", "conversation_id": str(voice_system.conversation_id)}


@router.post("/stop")
async def voice_stop(req: Request):
    """Stop the voice pipeline."""
    voice_system = getattr(req.app.state, "voice_system", None)
    if not voice_system:
        return {"error": "Voice not available"}

    if not voice_system.is_running:
        return {"status": "already_stopped"}

    await voice_system.stop()
    return {"status": "stopped"}


@router.post("/mode")
async def voice_set_mode(body: VoiceModeRequest, req: Request):
    """Change voice behavior mode."""
    from alchemy.voice.interface import VoiceMode

    voice_system = getattr(req.app.state, "voice_system", None)
    if not voice_system:
        return {"error": "Voice not available"}

    try:
        mode = VoiceMode(body.mode)
    except ValueError:
        return {"error": f"Invalid mode: {body.mode}", "valid": [m.value for m in VoiceMode]}

    voice_system.set_mode(mode)
    return {"status": "ok", "mode": mode.value}
