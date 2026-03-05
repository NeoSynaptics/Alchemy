"""Core module manifest — the agent kernel (Tier 0, locked)."""

from alchemy.manifest import ModelRequirement, ModuleManifest

MANIFEST = ModuleManifest(
    id="core",
    name="Agent Kernel",
    description="Playwright agent loop, LLM parsing, action execution, approval gates.",
    settings_prefix="pw_",
    enabled_key="pw_enabled",
    requires=["adapters"],
    tier="core",
    api_prefix="/v1",
    api_tags=["playwright"],
    models=[
        ModelRequirement(
            capability="reasoning",
            required=True,
            preferred_model="qwen3:14b",
            min_tier="warm",
        ),
        ModelRequirement(
            capability="vision",
            required=False,
            preferred_model="qwen2.5vl:7b",
            min_tier="warm",
            context_tokens=2048,
        ),
    ],
)
