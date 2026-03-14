"""NEO-N manifest — device file tunnel."""

from alchemy.manifest import ModuleManifest

MANIFEST = ModuleManifest(
    id="neo_n",
    name="NEO-N",
    description="Device file tunnel — receives files from phones/tablets via HTTP upload",
    settings_prefix="neo_n_",
    enabled_key="neo_n_enabled",
    requires=["connect"],
    tier="app",
    api_prefix="/v1/neo-n",
    api_tags=["neo-n"],
    models=[],  # Pure transport — no GPU models needed
)
