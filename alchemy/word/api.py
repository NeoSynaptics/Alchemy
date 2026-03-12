"""AlchemyWord API — text generation endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from alchemy.word.writer import VALID_MODES, generate

router = APIRouter(prefix="/word", tags=["word"])


class WordGenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input text to process")
    mode: str = Field(..., description="Generation mode: summarize, rewrite, expand, translate")
    model: str = Field(default="qwen3:14b", description="Ollama model to use")
    target_language: str = Field(default="English", description="Target language for translate mode")


class WordGenerateResponse(BaseModel):
    text: str
    mode: str
    model: str


@router.post("/generate", response_model=WordGenerateResponse)
async def generate_text(req: WordGenerateRequest, request: Request) -> WordGenerateResponse:
    """Generate text using an LLM (summarize, rewrite, expand, or translate)."""
    # Prefer APU gateway over raw OllamaClient
    _gw = getattr(request.app.state, "apu_gateway", None)
    ollama = _gw.with_caller("word", priority=2) if _gw else getattr(request.app.state, "ollama_client", None)
    if not ollama:
        raise HTTPException(status_code=503, detail="Ollama client not available")

    if req.mode not in VALID_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{req.mode}'. Must be one of: {sorted(VALID_MODES)}",
        )

    try:
        text = await generate(
            client=ollama,
            prompt=req.prompt,
            mode=req.mode,
            model=req.model,
            target_language=req.target_language,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return WordGenerateResponse(text=text, mode=req.mode, model=req.model)
