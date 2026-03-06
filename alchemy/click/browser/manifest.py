"""AlchemyBrowser manifest — Playwright-based web/Electron automation."""

from alchemy.manifest import ModelRequirement, ModuleManifest

MANIFEST = ModuleManifest(
    id="click.browser",
    name="AlchemyBrowser",
    description="Playwright accessibility tree agent. Ref-based actions via Qwen3 14B for web/Electron apps.",
    settings_prefix="click_browser_",
    requires=["adapters", "core"],
    tier="core",
    api_prefix="/v1",
    api_tags=["browser"],
    models=[
        ModelRequirement(
            capability="reasoning",
            required=True,
            preferred_model="qwen3:14b",
            min_tier="warm",
        ),
    ],
)
