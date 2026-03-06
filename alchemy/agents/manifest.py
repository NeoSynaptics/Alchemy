"""AlchemyAgents manifest — internal agent orchestration layer."""

from alchemy.manifest import ModelRequirement, ModuleManifest

MANIFEST = ModuleManifest(
    id="agents",
    name="AlchemyAgents",
    description=(
        "Internal agent orchestration. Houses agents that use "
        "AlchemyFlowAgent to automate specific targets (VS Code, etc.). "
        "All agents are internal-only, toggle on/off."
    ),
    settings_prefix="agents_",
    enabled_key="agents_enabled",
    requires=["click"],
    tier="core",
    api_prefix="/v1",
    api_tags=[],
    models=[
        ModelRequirement(
            capability="vision+clicking",
            required=True,
            preferred_model="qwen2.5vl:7b",
            min_tier="warm",
            context_tokens=2048,
        ),
    ],
)
