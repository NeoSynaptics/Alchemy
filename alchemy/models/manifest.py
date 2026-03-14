"""CPU model management manifest."""

from alchemy.manifest import ModuleManifest

MANIFEST = ModuleManifest(
    id="models",
    name="CPU Model Manager",
    description="CPU model lifecycle, health checks, and RAM scheduling",
    tier="infra",
    requires=["adapters"],
)
