"""GPU fleet manifest — VRAM/RAM orchestration and model registry."""

from alchemy.manifest import ModuleManifest

MANIFEST = ModuleManifest(
    id="gpu",
    name="GPU Fleet Orchestrator",
    description="VRAM/RAM orchestration, model registry, hot-swap, and memory budgets.",
    tier="core",
    api_prefix="/v1",
    api_tags=["gpu"],
)
