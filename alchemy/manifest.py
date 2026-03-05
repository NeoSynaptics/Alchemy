"""Module manifest — lightweight metadata for setup wizards, settings pages, and discovery.

Every Alchemy module that wants to be discoverable adds a manifest.py with:
    MANIFEST = ModuleManifest(id="mymod", name="My Module", ...)

The manifest is pure data — no imports from feature modules allowed.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelRequirement:
    """Declares that a module/app needs a specific model capability.

    The app tells the core WHAT it needs. The core decides WHERE to put it.

    Example:
        ModelRequirement(
            capability="vision",
            required=True,
            preferred_model="qwen2.5vl:7b",
            min_tier="warm",
        )
    """

    capability: str                     # "vision", "reasoning", "voice", "coding", "embedding"
    required: bool = True               # False = nice-to-have, app degrades gracefully without it
    preferred_model: str | None = None  # Hint: "qwen2.5vl:7b" — core may override
    min_tier: str = "warm"              # Minimum: "resident" | "warm" | "cold"
    context_tokens: int | None = None   # Hint: max context needed (e.g. 2048 for vision)


@dataclass(frozen=True)
class ModuleManifest:
    """Declares a module's identity, dependencies, and wizard metadata."""

    id: str                                     # "gate", "desktop", "cloud"
    name: str                                   # "Gate Reviewer"
    description: str                            # One sentence
    settings_prefix: str = ""                   # "gate_" — maps to settings group
    enabled_key: str | None = None              # "gate_enabled" — on/off toggle
    requires: list[str] = field(default_factory=list)   # ["ollama"] — module deps
    env_keys: list[str] = field(default_factory=list)   # ["ANTHROPIC_API_KEY"]
    tier: str = "app"                           # "core" | "infra" | "app"
    api_prefix: str | None = None               # "/gate" — if it has API routes
    api_tags: list[str] = field(default_factory=list)
    models: list[ModelRequirement] = field(default_factory=list)  # Model brain contract
