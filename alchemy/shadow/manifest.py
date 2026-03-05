"""Shadow desktop manifest — WSL2 hidden desktop for GUI automation."""

from alchemy.manifest import ModuleManifest

MANIFEST = ModuleManifest(
    id="shadow",
    name="Shadow Desktop",
    description="WSL2 Xvfb hidden desktop for headless GUI automation (Tier 2).",
    settings_prefix="shadow_",
    tier="infra",
    api_prefix="/v1",
    api_tags=["shadow"],
)
