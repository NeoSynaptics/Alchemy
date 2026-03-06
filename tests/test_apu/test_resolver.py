"""Tests for the model resolver — capability tags to actual model names."""

from __future__ import annotations

import pytest

from alchemy.apu.model_table import ALL_TAGS, COMBO_TABLE, SINGLE_TAG_TABLE
from alchemy.apu.registry import ModelBackend, ModelCard, ModelRegistry, ModelTier
from alchemy.apu.resolver import ModelResolver, ResolvedModel, ManifestResolution
from alchemy.manifest import ModelRequirement, ModuleManifest


def _make_registry(*names_and_caps: tuple[str, list[str]]) -> ModelRegistry:
    reg = ModelRegistry()
    for name, caps in names_and_caps:
        reg.register(ModelCard(
            name=name,
            display_name=name,
            backend=ModelBackend.OLLAMA,
            vram_mb=4096,
            ram_mb=4096,
            capabilities=caps,
            default_tier=ModelTier.WARM,
            current_tier=ModelTier.WARM,
        ))
    return reg


class TestResolverPinned:
    def test_pinned_model_available(self):
        reg = _make_registry(("qwen3:14b", ["reasoning"]))
        resolver = ModelResolver(reg)
        result = resolver.resolve(ModelRequirement(
            capability="reasoning",
            preferred_model="qwen3:14b",
        ))
        assert result.model_name == "qwen3:14b"
        assert result.resolution == "pinned"
        assert result.available is True

    def test_pinned_model_not_in_fleet_falls_back(self):
        reg = _make_registry(("qwen3:14b", ["reasoning"]))
        resolver = ModelResolver(reg)
        result = resolver.resolve(ModelRequirement(
            capability="reasoning",
            preferred_model="nonexistent-model",
        ))
        # Should fall back to auto-resolve since pinned not available
        assert result.model_name == "qwen3:14b"
        assert result.resolution == "pinned_fallback"
        assert result.available is True


class TestResolverSingleTag:
    def test_vision_resolves_to_vlm(self):
        reg = _make_registry(("qwen2.5vl:7b", ["vision"]))
        resolver = ModelResolver(reg)
        result = resolver.resolve(ModelRequirement(capability="vision"))
        assert result.model_name == "qwen2.5vl:7b"
        assert result.resolution == "single_tag"

    def test_reasoning_resolves(self):
        reg = _make_registry(("qwen3:14b", ["reasoning"]))
        resolver = ModelResolver(reg)
        result = resolver.resolve(ModelRequirement(capability="reasoning"))
        assert result.model_name == "qwen3:14b"
        assert result.resolution == "single_tag"

    def test_embedding_resolves(self):
        reg = _make_registry(("nomic-embed-text", ["embedding"]))
        resolver = ModelResolver(reg)
        result = resolver.resolve(ModelRequirement(capability="embedding"))
        assert result.model_name == "nomic-embed-text"

    def test_unknown_tag_unresolved(self):
        reg = _make_registry()
        resolver = ModelResolver(reg)
        result = resolver.resolve(ModelRequirement(capability="unknown_capability"))
        assert result.model_name is None
        assert result.resolution == "unresolved"


class TestResolverCombo:
    def test_vision_text_gets_vlm(self):
        reg = _make_registry(
            ("qwen2.5vl:7b", ["vision"]),
            ("qwen3:14b", ["reasoning"]),
        )
        resolver = ModelResolver(reg)
        result = resolver.resolve(ModelRequirement(capability="vision+text"))
        assert result.model_name == "qwen2.5vl:7b"
        assert result.resolution == "combo"

    def test_text_coding_gets_coder(self):
        reg = _make_registry(
            ("qwen2.5-coder:14b", ["coding"]),
            ("qwen3:14b", ["reasoning"]),
        )
        resolver = ModelResolver(reg)
        result = resolver.resolve(ModelRequirement(capability="text+coding"))
        assert result.model_name == "qwen2.5-coder:14b"
        assert result.resolution == "combo"

    def test_comma_separated_tags(self):
        reg = _make_registry(("qwen2.5vl:7b", ["vision"]))
        resolver = ModelResolver(reg)
        result = resolver.resolve(ModelRequirement(capability="vision,text"))
        assert result.model_name == "qwen2.5vl:7b"

    def test_space_separated_tags(self):
        reg = _make_registry(("qwen2.5vl:7b", ["vision"]))
        resolver = ModelResolver(reg)
        result = resolver.resolve(ModelRequirement(capability="vision text"))
        assert result.model_name == "qwen2.5vl:7b"


class TestResolverFallback:
    def test_falls_back_to_registry_capability(self):
        """If table doesn't match but registry has a model with the capability."""
        reg = _make_registry(("custom-model", ["custom_cap"]))
        resolver = ModelResolver(reg)
        result = resolver.resolve(ModelRequirement(capability="custom_cap"))
        assert result.model_name == "custom-model"
        assert result.resolution == "fallback"
        assert result.available is True


class TestManifestResolution:
    def test_full_manifest_resolution(self):
        reg = _make_registry(
            ("qwen3:14b", ["reasoning", "agent", "gate"]),
            ("qwen2.5vl:7b", ["vision", "escalation", "desktop"]),
        )
        manifest = ModuleManifest(
            id="test-app",
            name="Test App",
            description="Test",
            models=[
                ModelRequirement(capability="reasoning", required=True),
                ModelRequirement(capability="vision", required=False),
            ],
        )
        resolver = ModelResolver(reg)
        resolution = resolver.resolve_manifest(manifest)
        assert resolution.module_id == "test-app"
        assert resolution.all_resolved  # required met
        assert "qwen3:14b" in resolution.model_names
        assert "qwen2.5vl:7b" in resolution.model_names
        assert len(resolution.missing) == 0

    def test_missing_required_model(self):
        reg = _make_registry()  # Empty fleet
        manifest = ModuleManifest(
            id="test-app",
            name="Test App",
            description="Test",
            models=[
                ModelRequirement(capability="reasoning", required=True),
            ],
        )
        resolver = ModelResolver(reg)
        resolution = resolver.resolve_manifest(manifest)
        assert not resolution.all_resolved
        assert "reasoning" in resolution.missing

    def test_no_registry_assumes_available(self):
        """Testing mode: no registry = assume all models available."""
        resolver = ModelResolver(None)
        result = resolver.resolve(ModelRequirement(
            capability="reasoning",
            preferred_model="qwen3:14b",
        ))
        assert result.available is True
        assert result.model_name == "qwen3:14b"


class TestModelTable:
    def test_all_tags_complete(self):
        """Every tag in SINGLE_TAG_TABLE is in ALL_TAGS."""
        assert ALL_TAGS == frozenset(SINGLE_TAG_TABLE.keys())

    def test_combo_tags_are_subsets_of_all_tags(self):
        """All combo keys use valid tags."""
        for combo_tags in COMBO_TABLE:
            for tag in combo_tags:
                assert tag in ALL_TAGS, f"Combo tag '{tag}' not in ALL_TAGS"

    def test_single_tag_models_exist_in_fleet_yaml(self):
        """Spot check: key models referenced in tables should match fleet config."""
        key_models = {"qwen3:14b", "qwen2.5vl:7b", "nomic-embed-text", "whisper-large-v3"}
        all_model_names = set()
        for candidates in SINGLE_TAG_TABLE.values():
            for c in candidates:
                all_model_names.add(c.name)
        for model in key_models:
            assert model in all_model_names, f"{model} not referenced in any tag"

    def test_tag_count(self):
        """We should have a meaningful number of tags."""
        assert len(ALL_TAGS) >= 15
