"""AlchemyFlow manifest — vision-based desktop automation with ghost cursor."""

from alchemy.manifest import ModelRequirement, ModuleManifest

MANIFEST = ModuleManifest(
    id="click.flow",
    name="AlchemyFlow",
    description="Vision click agent: screenshot → Qwen2.5-VL 7B → coordinates → ghost cursor. For native Win32 apps.",
    settings_prefix="click_flow_",
    requires=["adapters", "core"],
    tier="core",
    api_prefix="/v1",
    api_tags=["vision"],
    models=[
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
