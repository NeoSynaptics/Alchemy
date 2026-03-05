"""Research manifest — AlchemyBrowser semantic search and synthesis."""

from alchemy.manifest import ModelRequirement, ModuleManifest

MANIFEST = ModuleManifest(
    id="research",
    name="AlchemyBrowser",
    description="Semantic web search with query decomposition, parallel fetch, and LLM synthesis.",
    settings_prefix="research_",
    enabled_key="research_enabled",
    requires=["adapters"],
    tier="app",
    api_prefix="/v1",
    api_tags=["research"],
    models=[
        ModelRequirement(
            capability="reasoning",
            required=True,
            preferred_model="qwen3:14b",
            min_tier="warm",
        ),
        ModelRequirement(
            capability="embedding",
            required=False,
            preferred_model="nomic-embed-text",
            min_tier="warm",
        ),
    ],
)
