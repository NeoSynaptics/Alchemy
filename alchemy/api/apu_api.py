"""APU (Alchemy Processing Unit) API — model placement, VRAM monitoring, app contracts."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from alchemy.apu.orchestrator import StackOrchestrator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/apu", tags=["apu"])


# --- Response models ---


class GPUProcessResponse(BaseModel):
    pid: int
    name: str
    vram_mb: int


class GPUInfoResponse(BaseModel):
    index: int
    name: str
    total_vram_mb: int
    used_vram_mb: int
    free_vram_mb: int
    temperature_c: int
    utilization_pct: int
    processes: list[GPUProcessResponse] = []


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
    app_priority: int = 50


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
        app_priority=card.app_priority,
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
            processes=[
                GPUProcessResponse(pid=p.pid, name=p.name, vram_mb=p.vram_mb)
                for p in getattr(g, "processes", [])
            ],
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
    from alchemy.apu.resolver import ModelResolver

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

    result = await orch.app_activate(app_name, resolution.model_names, module_tier=manifest.tier)
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
    from alchemy.apu.resolver import ModelResolver

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


# --- App Priority Management ---


class AppPriorityEntry(BaseModel):
    app_name: str
    priority: int  # 0=highest, 100=lowest
    models: list[str] = []  # models currently owned by this app
    gpu: int | None = None  # preferred GPU if set


class AppPriorityListResponse(BaseModel):
    apps: list[AppPriorityEntry]


class SetAppPriorityRequest(BaseModel):
    priority: int  # 0 = highest, 100 = lowest


class SetAppGPURequest(BaseModel):
    gpu: int | None  # 0, 1, or null for auto


class FrozenConfigResponse(BaseModel):
    gpu_0: list[str] = []
    gpu_1: list[str] = []
    ram: list[str] = []


class FrozenConfigRequest(BaseModel):
    gpu_0: list[str] = []
    gpu_1: list[str] = []
    ram: list[str] = []


class FrozenRestoreResponse(BaseModel):
    actions: list[str] = []


@router.get("/priority", response_model=AppPriorityListResponse)
async def list_app_priorities(request: Request) -> AppPriorityListResponse:
    """List all apps with their GPU priority ranking, sorted by priority (0=highest)."""
    orch = _get_orchestrator(request)
    prios = orch.all_app_priorities()
    app_models = orch._app_models
    entries = []
    for app_name, prio in prios.items():
        entries.append(AppPriorityEntry(
            app_name=app_name,
            priority=prio,
            models=app_models.get(app_name, []),
        ))
    return AppPriorityListResponse(apps=entries)


@router.post("/priority/{app_name}", response_model=AppPriorityEntry)
async def set_app_priority(
    request: Request, app_name: str, body: SetAppPriorityRequest,
) -> AppPriorityEntry:
    """Set an app's GPU priority. 0 = highest (models evicted last), 100 = lowest."""
    orch = _get_orchestrator(request)
    orch.set_app_priority(app_name, body.priority)
    return AppPriorityEntry(
        app_name=app_name,
        priority=orch.get_app_priority(app_name),
        models=orch._app_models.get(app_name, []),
    )


@router.post("/priority/{app_name}/gpu", response_model=AppPriorityEntry)
async def set_app_gpu(
    request: Request, app_name: str, body: SetAppGPURequest,
) -> AppPriorityEntry:
    """Set preferred GPU for an app. All its models will prefer this GPU."""
    orch = _get_orchestrator(request)
    if body.gpu is not None and body.gpu not in (0, 1):
        raise HTTPException(status_code=400, detail="GPU must be 0, 1, or null")
    orch.set_app_gpu(app_name, body.gpu)
    return AppPriorityEntry(
        app_name=app_name,
        priority=orch.get_app_priority(app_name),
        models=orch._app_models.get(app_name, []),
        gpu=body.gpu,
    )


# --- Frozen Baseline ---


@router.get("/frozen", response_model=FrozenConfigResponse)
async def get_frozen_config(request: Request) -> FrozenConfigResponse:
    """Get the current frozen baseline configuration."""
    orch = _get_orchestrator(request)
    config = orch.get_frozen_config()
    return FrozenConfigResponse(**config)


@router.post("/frozen", response_model=FrozenConfigResponse)
async def save_frozen_config(request: Request, body: FrozenConfigRequest) -> FrozenConfigResponse:
    """Save frozen baseline configuration. These models auto-load on boot and after task release."""
    orch = _get_orchestrator(request)
    orch.save_frozen_config(body.model_dump())
    return FrozenConfigResponse(**orch.get_frozen_config())


@router.post("/frozen/restore", response_model=FrozenRestoreResponse)
async def restore_frozen_baseline(request: Request) -> FrozenRestoreResponse:
    """Immediately restore the frozen baseline — load all configured models now."""
    orch = _get_orchestrator(request)
    actions = await orch.restore_frozen_baseline()
    return FrozenRestoreResponse(actions=actions)
