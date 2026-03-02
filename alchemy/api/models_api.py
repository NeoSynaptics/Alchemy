"""Models API — CPU model status.

NEO-TX (or other apps) can check what models Alchemy has loaded.
Stub (Phase 0) — returns mock data.
"""

from __future__ import annotations

from fastapi import APIRouter

from alchemy.schemas import ModelInfo, ModelsResponse

router = APIRouter(tags=["models"])


@router.get("/models", response_model=ModelsResponse)
async def get_models() -> ModelsResponse:
    """List loaded CPU models and RAM usage."""
    return ModelsResponse(
        models=[
            ModelInfo(name="ui-tars:72b", loaded=False, size_gb=42.0, ram_used_gb=None),
        ],
        total_ram_gb=128.0,
        available_ram_gb=86.0,
    )
