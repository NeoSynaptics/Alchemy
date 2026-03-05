"""Model Registry tests — tiers, eviction ordering, fleet config."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from alchemy.gpu.registry import (
    ModelBackend,
    ModelCard,
    ModelLocation,
    ModelRegistry,
    ModelTier,
)


def _card(name: str = "test-model", vram: int = 4096,
          tier: ModelTier = ModelTier.WARM,
          location: ModelLocation = ModelLocation.DISK,
          gpu: int | None = None,
          capabilities: list[str] | None = None) -> ModelCard:
    return ModelCard(
        name=name,
        display_name=name,
        backend=ModelBackend.OLLAMA,
        vram_mb=vram,
        ram_mb=vram,
        disk_mb=vram,
        preferred_gpu=gpu,
        default_tier=tier,
        current_tier=tier,
        current_location=location,
        capabilities=capabilities or [],
    )


def test_register_and_get():
    reg = ModelRegistry()
    card = _card("qwen3:14b")
    reg.register(card)
    assert reg.get("qwen3:14b") is card
    assert reg.get("nonexistent") is None


def test_unregister():
    reg = ModelRegistry()
    reg.register(_card("a"))
    reg.register(_card("b"))
    removed = reg.unregister("a")
    assert removed is not None
    assert removed.name == "a"
    assert reg.get("a") is None
    assert reg.get("b") is not None


def test_all_models():
    reg = ModelRegistry()
    reg.register(_card("a"))
    reg.register(_card("b"))
    reg.register(_card("c"))
    assert len(reg.all_models()) == 3


def test_update_location():
    reg = ModelRegistry()
    reg.register(_card("m"))
    assert reg.update_location("m", ModelLocation.GPU_0, ModelTier.AGENT)
    card = reg.get("m")
    assert card.current_location == ModelLocation.GPU_0
    assert card.current_tier == ModelTier.AGENT


def test_update_location_nonexistent():
    reg = ModelRegistry()
    assert not reg.update_location("nonexistent", ModelLocation.GPU_0)


def test_models_on_gpu():
    reg = ModelRegistry()
    reg.register(_card("a", location=ModelLocation.GPU_0))
    reg.register(_card("b", location=ModelLocation.GPU_1))
    reg.register(_card("c", location=ModelLocation.GPU_0))
    reg.register(_card("d", location=ModelLocation.CPU_RAM))

    on_0 = reg.models_on_gpu(0)
    assert len(on_0) == 2
    assert {m.name for m in on_0} == {"a", "c"}

    on_1 = reg.models_on_gpu(1)
    assert len(on_1) == 1
    assert on_1[0].name == "b"


def test_models_in_ram():
    reg = ModelRegistry()
    reg.register(_card("a", location=ModelLocation.CPU_RAM))
    reg.register(_card("b", location=ModelLocation.GPU_0))
    reg.register(_card("c", location=ModelLocation.CPU_RAM))

    in_ram = reg.models_in_ram()
    assert len(in_ram) == 2
    assert {m.name for m in in_ram} == {"a", "c"}


def test_models_by_tier():
    reg = ModelRegistry()
    reg.register(_card("a", tier=ModelTier.RESIDENT))
    reg.register(_card("b", tier=ModelTier.WARM))
    reg.register(_card("c", tier=ModelTier.RESIDENT))

    residents = reg.models_by_tier(ModelTier.RESIDENT)
    assert len(residents) == 2


def test_total_vram_on_gpu():
    reg = ModelRegistry()
    reg.register(_card("a", vram=4096, location=ModelLocation.GPU_0))
    reg.register(_card("b", vram=2048, location=ModelLocation.GPU_0))
    reg.register(_card("c", vram=8192, location=ModelLocation.GPU_1))

    assert reg.total_vram_on_gpu(0) == 6144
    assert reg.total_vram_on_gpu(1) == 8192


def test_find_by_capability():
    reg = ModelRegistry()
    reg.register(_card("a", capabilities=["voice", "stt"]))
    reg.register(_card("b", capabilities=["vision", "clicking"]))
    reg.register(_card("c", capabilities=["voice", "tts"]))

    voice = reg.find_by_capability("voice")
    assert len(voice) == 2
    assert {m.name for m in voice} == {"a", "c"}

    clicking = reg.find_by_capability("clicking")
    assert len(clicking) == 1


def test_eviction_candidates_excludes_residents():
    """P0 RESIDENT models are never eviction candidates."""
    reg = ModelRegistry()
    reg.register(_card("resident", tier=ModelTier.RESIDENT, location=ModelLocation.GPU_0))
    reg.register(_card("agent", tier=ModelTier.AGENT, location=ModelLocation.GPU_0))
    reg.register(_card("user", tier=ModelTier.USER_ACTIVE, location=ModelLocation.GPU_0))

    candidates = reg.eviction_candidates(0)
    names = [c.name for c in candidates]
    assert "resident" not in names
    assert "agent" in names
    assert "user" in names


def test_eviction_candidates_sorted_by_tier():
    """Higher tier number (more evictable) comes first."""
    reg = ModelRegistry()
    reg.register(_card("agent", tier=ModelTier.AGENT, location=ModelLocation.GPU_0))
    reg.register(_card("user", tier=ModelTier.USER_ACTIVE, location=ModelLocation.GPU_0))

    candidates = reg.eviction_candidates(0)
    # AGENT (priority=2) is more evictable than USER_ACTIVE (priority=1)
    assert candidates[0].name == "agent"
    assert candidates[1].name == "user"


def test_eviction_candidates_lru_within_tier():
    """Within same tier, least-recently-used first."""
    reg = ModelRegistry()

    old = _card("old", tier=ModelTier.AGENT, location=ModelLocation.GPU_0)
    old.last_used = datetime(2025, 1, 1, tzinfo=timezone.utc)

    new = _card("new", tier=ModelTier.AGENT, location=ModelLocation.GPU_0)
    new.last_used = datetime(2026, 1, 1, tzinfo=timezone.utc)

    reg.register(old)
    reg.register(new)

    candidates = reg.eviction_candidates(0)
    assert candidates[0].name == "old"  # Older = evicted first
    assert candidates[1].name == "new"


def test_model_card_touch():
    card = _card("test")
    assert card.last_used is None
    card.touch()
    assert card.last_used is not None
    assert card.last_used.tzinfo is not None  # Timezone-aware


def test_model_tier_priority():
    assert ModelTier.RESIDENT.priority == 0
    assert ModelTier.USER_ACTIVE.priority == 1
    assert ModelTier.AGENT.priority == 2
    assert ModelTier.WARM.priority == 3
    assert ModelTier.COLD.priority == 4


def test_model_location_is_gpu():
    assert ModelLocation.GPU_0.is_gpu
    assert ModelLocation.GPU_1.is_gpu
    assert not ModelLocation.CPU_RAM.is_gpu
    assert not ModelLocation.DISK.is_gpu


def test_model_location_gpu_index():
    assert ModelLocation.GPU_0.gpu_index == 0
    assert ModelLocation.GPU_1.gpu_index == 1
    assert ModelLocation.CPU_RAM.gpu_index is None


def test_load_fleet_config():
    """Fleet YAML is parsed into ModelCards."""
    yaml_content = """
models:
  - name: "qwen3:14b"
    display_name: "Qwen3 14B"
    backend: ollama
    vram_mb: 9216
    ram_mb: 9216
    disk_mb: 9216
    preferred_gpu: 1
    default_tier: resident
    capabilities: [conversation, agent]

  - name: "nomic-embed"
    display_name: "Nomic Embed"
    backend: ollama
    vram_mb: 512
    ram_mb: 512
    disk_mb: 512
    preferred_gpu: null
    default_tier: warm
    capabilities: [embedding]
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()

        reg = ModelRegistry()
        reg.load_fleet_config(f.name)

    assert len(reg.all_models()) == 2

    qwen = reg.get("qwen3:14b")
    assert qwen is not None
    assert qwen.display_name == "Qwen3 14B"
    assert qwen.backend == ModelBackend.OLLAMA
    assert qwen.vram_mb == 9216
    assert qwen.preferred_gpu == 1
    assert qwen.default_tier == ModelTier.RESIDENT
    assert qwen.current_tier == ModelTier.RESIDENT
    assert qwen.current_location == ModelLocation.DISK  # Not yet loaded
    assert "conversation" in qwen.capabilities

    nomic = reg.get("nomic-embed")
    assert nomic.default_tier == ModelTier.WARM
    assert nomic.preferred_gpu is None


def test_load_fleet_config_missing_file():
    """Missing config file is handled gracefully."""
    reg = ModelRegistry()
    reg.load_fleet_config("/nonexistent/path.yaml")
    assert len(reg.all_models()) == 0
