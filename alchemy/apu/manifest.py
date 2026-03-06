"""APU (Alchemy Processing Unit) manifest — VRAM/RAM orchestration and model registry."""

from alchemy.manifest import ModuleManifest

MANIFEST = ModuleManifest(
    id="apu",
    name="APU (Alchemy Processing Unit)",
    description="VRAM/RAM orchestration, model registry, hot-swap, memory budgets, and Alchemy health guard.",
    tier="core",
    api_prefix="/v1",
    api_tags=["apu"],
)
