"""Adapters manifest — concrete LLM backend implementations."""

from alchemy.manifest import ModuleManifest

MANIFEST = ModuleManifest(
    id="adapters",
    name="LLM Adapters",
    description="Concrete LLM implementations (Ollama, vLLM, GUI-Actor).",
    settings_prefix="ollama_",
    tier="infra",
)
