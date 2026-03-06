"""AlchemyClick manifest — parent behavior contract for GUI automation.

AlchemyClick is the umbrella. It declares both model requirements
(reasoning for AlchemyBrowser, vision for AlchemyFlow) and owns
the shared behavior: task lifecycle, approval gates, tier classification.
"""

from alchemy.manifest import ModelRequirement, ModuleManifest

MANIFEST = ModuleManifest(
    id="click",
    name="AlchemyClick",
    description=(
        "GUI automation behavior contract. "
        "AlchemyBrowser (Playwright + Qwen3 14B) for web/Electron. "
        "AlchemyFlow (Qwen2.5-VL 7B + ghost cursor) for native desktop."
    ),
    settings_prefix="click_",
    enabled_key="click_enabled",
    requires=["adapters", "core"],
    tier="core",
    api_prefix="/v1",
    api_tags=["vision", "browser"],
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
        ModelRequirement(
            capability="ui-element-detection",
            required=False,
            preferred_model="omniparser-v2",
            min_tier="warm",
        ),
    ],
)
