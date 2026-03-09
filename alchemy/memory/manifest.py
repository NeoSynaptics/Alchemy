"""AlchemyMemory manifest — two-layer persistent memory + AI-native search UI."""

from alchemy.manifest import ModelRequirement, ModuleManifest

MANIFEST = ModuleManifest(
    id="memory",
    name="AlchemyMemory",
    description=(
        "Two-layer persistent memory: long-term timeline (screenshots + events, never erased) "
        "with semantic search, and short-term cache that signals the APU which models to keep warm. "
        "Includes a unified search UI that combines memory, context, and internet."
    ),
    settings_prefix="memory_",
    enabled_key="memory_enabled",
    requires=["adapters", "apu"],
    tier="infra",
    api_prefix="/v1/memory",
    api_tags=["memory"],
    models=[
        ModelRequirement(
            capability="vision",
            required=True,
            preferred_model="qwen2.5vl:7b",
            min_tier="warm",
            context_tokens=8192,
        ),
        ModelRequirement(
            capability="embedding",
            required=True,
            preferred_model="nomic-embed-text",
            min_tier="warm",
        ),
        ModelRequirement(
            capability="reasoning",
            required=True,
            preferred_model="qwen3:14b",
            min_tier="warm",
        ),
        ModelRequirement(
            capability="conversation",
            required=True,
            preferred_model="qwen3:3b",
            min_tier="warm",
        ),
    ],
)
