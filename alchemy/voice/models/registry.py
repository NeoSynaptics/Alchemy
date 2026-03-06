"""Model registry — register models by capability, look up defaults."""

from __future__ import annotations

import logging

from alchemy.voice.models.schemas import ModelCapability, ModelCard

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry of available models and their capabilities.

    Thread-safe for reads (dict lookups). Write operations (register/unregister)
    happen at startup only.
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelCard] = {}
        self._defaults: dict[ModelCapability, str] = {}

    def register(self, card: ModelCard) -> None:
        """Register a model. If it declares is_default_for, set it as default."""
        self._models[card.name] = card
        for cap in card.is_default_for:
            self._defaults[cap] = card.name
        logger.info(
            "Registered model: %s (caps=%s, speed=%s, endpoint=%s)",
            card.name,
            [c.value for c in card.capabilities],
            card.speed_tier.value,
            card.endpoint or "default",
        )

    def unregister(self, name: str) -> None:
        """Remove a model from the registry."""
        card = self._models.pop(name, None)
        if card:
            self._defaults = {k: v for k, v in self._defaults.items() if v != name}

    def get(self, name: str) -> ModelCard | None:
        return self._models.get(name)

    def get_default(self, capability: ModelCapability) -> ModelCard | None:
        name = self._defaults.get(capability)
        return self._models.get(name) if name else None

    def find_by_capability(self, capability: ModelCapability) -> list[ModelCard]:
        return [c for c in self._models.values() if capability in c.capabilities]

    def all_models(self) -> list[ModelCard]:
        return list(self._models.values())


def build_default_registry() -> ModelRegistry:
    """Create registry with the standard AlchemyVoice model set. Called at startup."""
    from config.settings import settings

    registry = ModelRegistry()

    # 14B conversational — GPU resident, fast
    registry.register(
        ModelCard(
            name=settings.gpu_model,
            capabilities=[ModelCapability.CONVERSATION, ModelCapability.CLASSIFICATION],
            speed_tier="fast",
            location="gpu_local",
            endpoint=settings.ollama_host,
            vram_gb=9.0,
            keep_alive=settings.gpu_model_keep_alive,
            max_context_tokens=32768,
            is_default_for=[ModelCapability.CONVERSATION, ModelCapability.CLASSIFICATION],
        )
    )

    # UI-TARS-72B — CPU via Alchemy, slow
    registry.register(
        ModelCard(
            name="ui-tars:72b",
            capabilities=[ModelCapability.VISION],
            speed_tier="slow",
            location="cpu_remote",
            endpoint=settings.alchemy_host,
            ram_gb=42.0,
            max_context_tokens=8192,
            is_default_for=[ModelCapability.VISION],
        )
    )

    return registry
