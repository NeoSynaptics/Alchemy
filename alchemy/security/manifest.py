"""Security module manifest — bearer token authentication."""

from alchemy.manifest import ModuleManifest

MANIFEST = ModuleManifest(
    id="security",
    name="Security",
    description="Bearer token authentication middleware for API routes.",
    tier="infra",
    settings_prefix="auth_",
    enabled_key="security_enabled",
    env_keys=["ALCHEMY_API_TOKEN"],
)
