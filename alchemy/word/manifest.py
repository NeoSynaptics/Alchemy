"""Word manifest -- AlchemyWord AI-native text editor."""

from alchemy.manifest import ModelRequirement, ModuleManifest

MANIFEST = ModuleManifest(
    id="word",
    name="AlchemyWord",
    description="AI-native text editor with inline completion, annotation, and style adaptation.",
    settings_prefix="word_",
    enabled_key="word_enabled",
    requires=["adapters"],
    tier="app",
    api_prefix="/v1",
    api_tags=["word"],
    models=[
        ModelRequirement(
            capability="reasoning",
            required=True,
            min_tier="warm",
        ),
    ],
)
