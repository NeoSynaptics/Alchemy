"""AlchemyAssets manifest — client-side SDK packages."""

from alchemy.manifest import ModuleManifest

MANIFEST = ModuleManifest(
    id="assets",
    name="AlchemyAssets",
    description=(
        "Client-side SDK packages for apps that connect to Alchemy. "
        "Framework-agnostic, publishable as npm packages. "
        "Includes: @alchemy/connect (WebSocket client, text + binary)."
    ),
    settings_prefix="assets_",
    enabled_key="assets_enabled",
    requires=[],
    tier="infra",
    api_prefix=None,
    api_tags=[],
    models=[],  # Client SDKs — no GPU models
)
