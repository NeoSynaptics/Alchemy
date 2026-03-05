"""Model registry with priority tiers and GPU placement tracking."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    """Priority tiers for VRAM allocation. Lower number = higher priority."""

    RESIDENT = "resident"  # P0 — never evicted (voice + GUI clicker)
    USER_ACTIVE = "user_active"  # P1 — evicted only by P0
    AGENT = "agent"  # P2 — evicted by P0 or P1
    WARM = "warm"  # P3 — in RAM, not VRAM
    COLD = "cold"  # P4 — on disk only

    @property
    def priority(self) -> int:
        """Lower = higher priority (harder to evict)."""
        return _TIER_PRIORITY[self]


_TIER_PRIORITY = {
    ModelTier.RESIDENT: 0,
    ModelTier.USER_ACTIVE: 1,
    ModelTier.AGENT: 2,
    ModelTier.WARM: 3,
    ModelTier.COLD: 4,
}


class ModelLocation(str, Enum):
    GPU_0 = "gpu_0"
    GPU_1 = "gpu_1"
    CPU_RAM = "cpu_ram"
    DISK = "disk"
    NOT_DOWNLOADED = "not_downloaded"

    @property
    def is_gpu(self) -> bool:
        return self in (ModelLocation.GPU_0, ModelLocation.GPU_1)

    @property
    def gpu_index(self) -> int | None:
        if self == ModelLocation.GPU_0:
            return 0
        if self == ModelLocation.GPU_1:
            return 1
        return None


class ModelBackend(str, Enum):
    OLLAMA = "ollama"
    VLLM = "vllm"
    SUBPROCESS = "subprocess"


class ModelCard(BaseModel):
    """A model in the fleet with its placement and priority metadata."""

    name: str
    display_name: str = ""
    backend: ModelBackend = ModelBackend.OLLAMA
    vram_mb: int = 0
    ram_mb: int = 0
    disk_mb: int = 0
    preferred_gpu: int | None = None
    default_tier: ModelTier = ModelTier.WARM
    current_tier: ModelTier = ModelTier.COLD
    current_location: ModelLocation = ModelLocation.DISK
    capabilities: list[str] = Field(default_factory=list)
    last_used: datetime | None = None
    owner_app: str | None = None

    def touch(self) -> None:
        """Update last_used timestamp."""
        self.last_used = datetime.now(timezone.utc)


class ModelRegistry:
    """Thread-safe model registry with tier-aware eviction ordering."""

    def __init__(self) -> None:
        self._models: dict[str, ModelCard] = {}

    def register(self, card: ModelCard) -> None:
        self._models[card.name] = card
        logger.debug("Registered model: %s (tier=%s)", card.name, card.default_tier)

    def unregister(self, name: str) -> ModelCard | None:
        return self._models.pop(name, None)

    def get(self, name: str) -> ModelCard | None:
        return self._models.get(name)

    def all_models(self) -> list[ModelCard]:
        return list(self._models.values())

    def update_location(
        self, name: str, location: ModelLocation, tier: ModelTier | None = None
    ) -> bool:
        card = self._models.get(name)
        if card is None:
            return False
        card.current_location = location
        if tier is not None:
            card.current_tier = tier
        return True

    def models_on_gpu(self, gpu_index: int) -> list[ModelCard]:
        loc = ModelLocation.GPU_0 if gpu_index == 0 else ModelLocation.GPU_1
        return [m for m in self._models.values() if m.current_location == loc]

    def models_in_ram(self) -> list[ModelCard]:
        return [m for m in self._models.values() if m.current_location == ModelLocation.CPU_RAM]

    def models_by_tier(self, tier: ModelTier) -> list[ModelCard]:
        return [m for m in self._models.values() if m.current_tier == tier]

    def total_vram_on_gpu(self, gpu_index: int) -> int:
        return sum(m.vram_mb for m in self.models_on_gpu(gpu_index))

    def find_by_capability(self, capability: str) -> list[ModelCard]:
        return [m for m in self._models.values() if capability in m.capabilities]

    def eviction_candidates(self, gpu_index: int) -> list[ModelCard]:
        """Models on this GPU sorted for eviction: highest tier priority (most evictable) first,
        then least-recently-used within same tier. P0 RESIDENT models are excluded."""
        on_gpu = self.models_on_gpu(gpu_index)
        evictable = [m for m in on_gpu if m.current_tier != ModelTier.RESIDENT]
        return sorted(
            evictable,
            key=lambda m: (
                -m.current_tier.priority,  # Higher priority number = more evictable = first
                m.last_used or datetime.min.replace(tzinfo=timezone.utc),  # Older = first
            ),
        )

    def load_fleet_config(self, path: str | Path) -> None:
        """Load model fleet from a YAML config file."""
        import yaml

        path = Path(path)
        if not path.exists():
            logger.warning("Fleet config not found: %s", path)
            return

        with open(path) as f:
            data = yaml.safe_load(f)

        if not data or "models" not in data:
            logger.warning("Fleet config has no 'models' key: %s", path)
            return

        for entry in data["models"]:
            card = ModelCard(
                name=entry["name"],
                display_name=entry.get("display_name", entry["name"]),
                backend=ModelBackend(entry.get("backend", "ollama")),
                vram_mb=entry.get("vram_mb", 0),
                ram_mb=entry.get("ram_mb", 0),
                disk_mb=entry.get("disk_mb", 0),
                preferred_gpu=entry.get("preferred_gpu"),
                default_tier=ModelTier(entry.get("default_tier", "cold")),
                current_tier=ModelTier(entry.get("default_tier", "cold")),
                current_location=ModelLocation.DISK,
                capabilities=entry.get("capabilities", []),
            )
            self.register(card)

        logger.info("Loaded %d models from fleet config: %s", len(data["models"]), path)
