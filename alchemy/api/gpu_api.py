"""GPU Stack Orchestrator API — model placement, VRAM monitoring, app contracts."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from alchemy.gpu.orchestrator import StackOrchestrator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/gpu", tags=["gpu"])


# --- Response models ---


class GPUInfoResponse(BaseModel):
    index: int
    name: str
    total_vram_mb: int
    used_vram_mb: int
    free_vram_mb: int
    temperature_c: int
    utilization_pct: int


class RAMInfoResponse(BaseModel):
    total_mb: int
    used_mb: int
    free_mb: int
    available_mb: int


class ModelCardResponse(BaseModel):
    name: str
    display_name: str
    backend: str
    vram_mb: int
    ram_mb: int
    disk_mb: int
    preferred_gpu: int | None
    default_tier: str
    current_tier: str
    current_location: str
    capabilities: list[str]
    last_used: str | None
    owner_app: str | None


class StackStatusResponse(BaseModel):
    gpus: list[GPUInfoResponse]
    ram: RAMInfoResponse
    models: list[ModelCardResponse]
    mode: str


class LoadResponse(BaseModel):
    success: bool
    location: str | None = None
    evicted: list[str] = []
    error: str | None = None


class AppActivateRequest(BaseModel):
    models: list[str]


class RebalanceResponse(BaseModel):
    actions: list[str]


class DemotedResponse(BaseModel):
    success: bool
    demoted: list[str] = []


class ResolvedModelInfo(BaseModel):
    capability: str
    model_name: str | None = None
    resolution: str  # "pinned" | "combo" | "single_tag" | "fallback" | "unresolved"
    available: bool = False
    candidates: list[str] = []


class ManifestActivateResponse(BaseModel):
    success: bool
    resolved: list[ResolvedModelInfo] = []
    loaded: list[str] = []
    error: str | None = None


# --- Helpers ---


def _get_orchestrator(request: Request) -> StackOrchestrator:
    orch = getattr(request.app.state, "orchestrator", None)
    if orch is None:
        raise HTTPException(status_code=503, detail="GPU orchestrator not initialized")
    return orch


def _model_to_response(card: Any) -> ModelCardResponse:
    return ModelCardResponse(
        name=card.name,
        display_name=card.display_name,
        backend=card.backend.value,
        vram_mb=card.vram_mb,
        ram_mb=card.ram_mb,
        disk_mb=card.disk_mb,
        preferred_gpu=card.preferred_gpu,
        default_tier=card.default_tier.value,
        current_tier=card.current_tier.value,
        current_location=card.current_location.value,
        capabilities=card.capabilities,
        last_used=card.last_used.isoformat() if card.last_used else None,
        owner_app=card.owner_app,
    )


# --- Endpoints ---


@router.get("/status", response_model=StackStatusResponse)
async def gpu_stack_status(request: Request) -> StackStatusResponse:
    """Full system status: GPUs + RAM + all model placements."""
    orch = _get_orchestrator(request)
    status = await orch.status()

    gpus = [
        GPUInfoResponse(
            index=g.index,
            name=g.name,
            total_vram_mb=g.total_vram_mb,
            used_vram_mb=g.used_vram_mb,
            free_vram_mb=g.free_vram_mb,
            temperature_c=g.temperature_c,
            utilization_pct=g.utilization_pct,
        )
        for g in status.snapshot.gpus
    ]

    ram = RAMInfoResponse(
        total_mb=status.snapshot.ram.total_mb,
        used_mb=status.snapshot.ram.used_mb,
        free_mb=status.snapshot.ram.free_mb,
        available_mb=status.snapshot.ram.available_mb,
    )

    models = [_model_to_response(m) for m in status.models]

    return StackStatusResponse(gpus=gpus, ram=ram, models=models, mode=status.mode)


@router.get("/gpus", response_model=list[GPUInfoResponse])
async def list_gpus(request: Request) -> list[GPUInfoResponse]:
    """Raw GPU info only."""
    orch = _get_orchestrator(request)
    gpus = await orch.gpu_status()
    return [
        GPUInfoResponse(
            index=g.index, name=g.name,
            total_vram_mb=g.total_vram_mb, used_vram_mb=g.used_vram_mb,
            free_vram_mb=g.free_vram_mb, temperature_c=g.temperature_c,
            utilization_pct=g.utilization_pct,
        )
        for g in gpus
    ]


@router.get("/ram", response_model=RAMInfoResponse)
async def get_ram(request: Request) -> RAMInfoResponse:
    """System RAM info."""
    orch = _get_orchestrator(request)
    ram = await orch.ram_status()
    return RAMInfoResponse(
        total_mb=ram.total_mb, used_mb=ram.used_mb,
        free_mb=ram.free_mb, available_mb=ram.available_mb,
    )


@router.get("/models", response_model=list[ModelCardResponse])
async def list_models(request: Request) -> list[ModelCardResponse]:
    """All registered models with their current placement."""
    orch = _get_orchestrator(request)
    status = await orch.status()
    return [_model_to_response(m) for m in status.models]


@router.get("/models/{name}", response_model=ModelCardResponse)
async def get_model(request: Request, name: str) -> ModelCardResponse:
    """Get a single model's status."""
    orch = _get_orchestrator(request)
    card = orch._registry.get(name)
    if card is None:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    return _model_to_response(card)


@router.post("/models/{name}/promote", response_model=LoadResponse)
async def promote_model(request: Request, name: str) -> LoadResponse:
    """Promote a model from RAM/disk → VRAM."""
    orch = _get_orchestrator(request)
    result = await orch.promote(name)
    return LoadResponse(
        success=result.success,
        location=result.location.value if result.location else None,
        evicted=result.evicted,
        error=result.error,
    )


@router.post("/models/{name}/demote", response_model=LoadResponse)
async def demote_model(request: Request, name: str) -> LoadResponse:
    """Demote a model from VRAM → RAM (stays warm)."""
    orch = _get_orchestrator(request)
    ok = await orch.demote(name)
    return LoadResponse(success=ok, error=None if ok else "Demote failed or model is P0 resident")


@router.post("/models/{name}/load", response_model=LoadResponse)
async def load_model(request: Request, name: str) -> LoadResponse:
    """Load a model to VRAM (full load from any state)."""
    orch = _get_orchestrator(request)
    result = await orch.ensure_loaded(name)
    return LoadResponse(
        success=result.success,
        location=result.location.value if result.location else None,
        evicted=result.evicted,
        error=result.error,
    )


@router.post("/models/{name}/unload", response_model=LoadResponse)
async def unload_model(request: Request, name: str) -> LoadResponse:
    """Unload a model to disk (cold storage)."""
    orch = _get_orchestrator(request)
    ok = await orch.unload_model(name)
    return LoadResponse(success=ok, error=None if ok else "Unload failed or model is P0 resident")


@router.post("/rebalance", response_model=RebalanceResponse)
async def rebalance(request: Request) -> RebalanceResponse:
    """Re-evaluate all model placements and move misplaced models."""
    orch = _get_orchestrator(request)
    actions = await orch.rebalance()
    return RebalanceResponse(actions=actions)


@router.post("/app/{app_name}/activate", response_model=LoadResponse)
async def app_activate(request: Request, app_name: str, body: AppActivateRequest) -> LoadResponse:
    """Activate models for an app. Marks them P1 USER_ACTIVE."""
    orch = _get_orchestrator(request)
    result = await orch.app_activate(app_name, body.models)
    return LoadResponse(
        success=result.success,
        error=result.error,
    )


@router.post("/app/{app_name}/activate-manifest", response_model=ManifestActivateResponse)
async def app_activate_manifest(request: Request, app_name: str) -> ManifestActivateResponse:
    """Activate models for an app using its registered manifest.

    Reads the app's manifest from the module registry, resolves capability
    tags to actual model names via the internal model table, then loads
    them to VRAM.

    This is the preferred activation path — apps don't need to know model names.
    """
    from alchemy.registry import get as get_manifest
    from alchemy.gpu.resolver import ModelResolver

    manifest = get_manifest(app_name)
    if manifest is None:
        raise HTTPException(
            status_code=404,
            detail=f"Module '{app_name}' not found in registry. Add a manifest.py first.",
        )

    orch = _get_orchestrator(request)
    resolver = ModelResolver(orch._registry)
    resolution = resolver.resolve_manifest(manifest)

    # Show what was resolved
    resolved_details = [
        ResolvedModelInfo(
            capability=r.requirement.capability,
            model_name=r.model_name,
            resolution=r.resolution,
            available=r.available,
            candidates=r.candidates,
        )
        for r in resolution.models
    ]

    if not resolution.model_names:
        missing = resolution.missing
        if missing:
            return ManifestActivateResponse(
                success=False,
                resolved=resolved_details,
                error=f"No models for required capabilities: {missing}",
            )
        return ManifestActivateResponse(success=True, resolved=resolved_details)

    result = await orch.app_activate(app_name, resolution.model_names)
    return ManifestActivateResponse(
        success=result.success,
        resolved=resolved_details,
        loaded=resolution.model_names,
        error=result.error,
    )


@router.post("/resolve/{module_id}", response_model=list[ResolvedModelInfo])
async def resolve_models(request: Request, module_id: str) -> list[ResolvedModelInfo]:
    """Dry-run: resolve a module's capabilities to model names without loading.

    Use this to preview what models would be picked for an app.
    """
    from alchemy.registry import get as get_manifest
    from alchemy.gpu.resolver import ModelResolver

    manifest = get_manifest(module_id)
    if manifest is None:
        raise HTTPException(
            status_code=404,
            detail=f"Module '{module_id}' not found in registry",
        )

    orch = _get_orchestrator(request)
    resolver = ModelResolver(orch._registry)
    resolution = resolver.resolve_manifest(manifest)

    return [
        ResolvedModelInfo(
            capability=r.requirement.capability,
            model_name=r.model_name,
            resolution=r.resolution,
            available=r.available,
            candidates=r.candidates,
        )
        for r in resolution.models
    ]


@router.post("/app/{app_name}/deactivate", response_model=DemotedResponse)
async def app_deactivate(request: Request, app_name: str) -> DemotedResponse:
    """Deactivate an app. Demote its models to WARM."""
    orch = _get_orchestrator(request)
    demoted = await orch.app_deactivate(app_name)
    return DemotedResponse(success=True, demoted=demoted)
