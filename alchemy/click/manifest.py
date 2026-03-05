"""AlchemyClick manifest — two-tier GUI automation (Playwright + vision fallback)."""

from alchemy.manifest import ModelRequirement, ModuleManifest

MANIFEST = ModuleManifest(
    id="click",
    name="AlchemyClick",
    description="Two-tier GUI click agent: Playwright a11y tree + Qwen2.5-VL vision fallback.",
    settings_prefix="click_",
    enabled_key="click_enabled",
    requires=["adapters", "core"],
    tier="core",
    api_prefix="/v1",
    api_tags=["vision"],
    models=[
        ModelRequirement(
            capability="reasoning",
            required=True,
            preferred_model="qwen3:14b",
            min_tier="warm",
        ),
        ModelRequirement(
            capability="vision+clicking",
            required=True,
            preferred_model="qwen2.5vl:7b",
            min_tier="warm",
            context_tokens=2048,
        ),
    ],
)
