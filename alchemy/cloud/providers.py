"""Cloud AI provider registry.

Each provider defines what it needs (API key, extension, etc.)
and how to validate the credentials work.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CloudProvider:
    """A cloud AI provider that Alchemy can set up for the user."""
    id: str  # e.g. "claude", "openai", "gemini"
    name: str  # e.g. "Claude (Anthropic)"
    vscode_extension: str | None  # Extension ID to install, e.g. "anthropic.claude-code"
    env_key: str  # Environment variable for the API key
    validate_url: str | None  # URL to ping to check if key works
    setup_instructions: str  # Human-readable setup help
    default: bool = False  # Is this the recommended provider?
    extra_config: dict = field(default_factory=dict)


_PROVIDERS: dict[str, CloudProvider] = {
    "claude": CloudProvider(
        id="claude",
        name="Claude (Anthropic)",
        vscode_extension="anthropic.claude-code",
        env_key="ANTHROPIC_API_KEY",
        validate_url="https://api.anthropic.com/v1/messages",
        setup_instructions="Get your API key at console.anthropic.com/settings/keys",
        default=True,
    ),
    "openai": CloudProvider(
        id="openai",
        name="ChatGPT / OpenAI",
        vscode_extension=None,  # Multiple options, user picks
        env_key="OPENAI_API_KEY",
        validate_url="https://api.openai.com/v1/models",
        setup_instructions="Get your API key at platform.openai.com/api-keys",
    ),
    "gemini": CloudProvider(
        id="gemini",
        name="Gemini (Google)",
        vscode_extension=None,
        env_key="GOOGLE_API_KEY",
        validate_url=None,
        setup_instructions="Get your API key at aistudio.google.com/apikey",
    ),
}


def get_provider(provider_id: str) -> CloudProvider | None:
    """Get a provider by ID."""
    return _PROVIDERS.get(provider_id)


def list_providers() -> list[CloudProvider]:
    """List all available providers, default first."""
    return sorted(_PROVIDERS.values(), key=lambda p: (not p.default, p.name))
