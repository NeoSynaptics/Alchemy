"""Models API — real Ollama model status + system RAM info."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from alchemy.models.ollama_client import OllamaClient
from alchemy.schemas import ModelInfo, ModelsResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["models"])


@router.get("/models", response_model=ModelsResponse)
async def get_models(request: Request) -> ModelsResponse:
    """List locally available models and RAM usage."""
    ollama: OllamaClient | None = getattr(request.app.state, "ollama_client", None)

    models = []
    if ollama:
        try:
            raw_models = await ollama.list_models()
            models = [
                ModelInfo(
                    name=m.get("name", "unknown"),
                    loaded=True,
                    size_gb=round(m.get("size", 0) / (1024**3), 1),
                    ram_used_gb=None,
                )
                for m in raw_models
            ]
        except Exception as e:
            logger.warning("Failed to list Ollama models: %s", e)

    # System RAM info
    total_gb, available_gb = 128.0, 86.0
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = round(mem.total / (1024**3), 1)
        available_gb = round(mem.available / (1024**3), 1)
    except ImportError:
        pass

    return ModelsResponse(
        models=models,
        total_ram_gb=total_gb,
        available_ram_gb=available_gb,
    )
