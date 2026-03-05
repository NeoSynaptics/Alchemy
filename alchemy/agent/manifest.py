"""Agent manifest — two-tier GUI automation (Playwright + vision fallback)."""

from alchemy.manifest import ModelRequirement, ModuleManifest

MANIFEST = ModuleManifest(
    id="agent",
    name="GUI Agent",
    description="Two-tier GUI automation: Playwright a11y tree + Qwen2.5-VL vision fallback.",
    requires=["adapters", "core"],
    tier="app",
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
