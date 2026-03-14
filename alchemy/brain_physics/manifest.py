"""BrainPhysics manifest — cognitive routing via coarse-to-fine physics simulation."""

from alchemy.manifest import ModelRequirement, ModuleManifest

MANIFEST = ModuleManifest(
    id="brain_physics",
    name="BrainPhysics",
    description="Coarse-to-fine cognitive router: spatial scene graphs, intuitive physics simulation, and predictive processing loop.",
    settings_prefix="brain_physics_",
    enabled_key="brain_physics_enabled",
    requires=["adapters"],
    tier="app",
    api_prefix="/v1",
    api_tags=["brain_physics"],
    models=[
        # Vision: coarse-to-fine scene perception (extract objects + spatial relations)
        ModelRequirement(
            capability="vision",
            required=True,
            preferred_model="qwen2.5vl:7b",
            min_tier="warm",
            context_tokens=2048,
        ),
        # Reasoning: prediction refinement loop + consolidation
        ModelRequirement(
            capability="reasoning",
            required=True,
            preferred_model="qwen3:14b",
            min_tier="warm",
        ),
        # Embedding: memory consolidation / distilled knowledge lookup
        ModelRequirement(
            capability="embedding",
            required=False,
            preferred_model="nomic-embed-text",
            min_tier="warm",
        ),
    ],
)
