"""AlchemyConnect manifest — universal tunnel/bus for external apps."""

from alchemy.manifest import ModuleManifest

MANIFEST = ModuleManifest(
    id="connect",
    name="AlchemyConnect",
    description=(
        "Universal tunnel/bus for external apps to reach Alchemy. "
        "WebSocket transport with QR-locked device pairing and "
        "agent-based message routing."
    ),
    settings_prefix="connect_",
    enabled_key="connect_enabled",
    requires=[],
    tier="infra",
    api_prefix="/ws",
    api_tags=["connect"],
    models=[],  # Pure transport — no GPU models needed
)
