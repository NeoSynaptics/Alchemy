"""Cloud AI Bridge manifest — provider setup and credential management."""

from alchemy.manifest import ModuleManifest

MANIFEST = ModuleManifest(
    id="cloud",
    name="Cloud AI Bridge",
    description="Cloud AI provider setup, API key storage, and VS Code extension management.",
    tier="core",
    env_keys=["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"],
)
