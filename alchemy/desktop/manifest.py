"""Desktop agent manifest — native Windows automation with ghost cursor."""

from alchemy.manifest import ModelRequirement, ModuleManifest

MANIFEST = ModuleManifest(
    id="desktop",
    name="Desktop Agent",
    description="Vision-driven native Windows automation with ghost cursor and SendInput.",
    settings_prefix="desktop_",
    enabled_key="desktop_enabled",
    requires=["adapters"],
    tier="core",
    api_prefix="/v1",
    api_tags=["desktop"],
    models=[
        ModelRequirement(
            capability="vision",
            required=True,
            preferred_model="qwen2.5vl:7b",
            min_tier="warm",
            context_tokens=2048,
        ),
    ],
)
