"""Router manifest — context injection and task classification."""

from alchemy.manifest import ModelRequirement, ModuleManifest

MANIFEST = ModuleManifest(
    id="router",
    name="Context Router",
    description="Task classification, environment detection, and context injection for agents.",
    settings_prefix="router_",
    enabled_key="router_enabled",
    tier="infra",
    models=[
        ModelRequirement(
            capability="classification",
            required=False,
            preferred_model="deberta-v3-router",
            min_tier="warm",
        ),
    ],
)
