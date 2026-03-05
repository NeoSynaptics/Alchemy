"""Gate reviewer manifest — Claude Code tool call auto-approve/deny."""

from alchemy.manifest import ModelRequirement, ModuleManifest

MANIFEST = ModuleManifest(
    id="gate",
    name="Gate Reviewer",
    description="Claude Code tool call reviewer via Qwen3 14B (auto-approve/deny).",
    settings_prefix="gate_",
    enabled_key="gate_enabled",
    requires=["adapters"],
    tier="app",
    api_prefix="/gate",
    api_tags=["gate"],
    models=[
        ModelRequirement(
            capability="reasoning",
            required=True,
            preferred_model="qwen3:14b",
            min_tier="warm",
        ),
    ],
)
