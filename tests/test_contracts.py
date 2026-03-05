"""Tests for model contract validation."""

from __future__ import annotations

from alchemy.contracts import (
    ContractReport,
    RequirementResult,
    validate_contracts,
    validate_module_contract,
)
from alchemy.gpu.registry import ModelBackend, ModelCard, ModelRegistry, ModelTier
from alchemy.manifest import ModelRequirement, ModuleManifest


def _make_registry(*cards: ModelCard) -> ModelRegistry:
    reg = ModelRegistry()
    for card in cards:
        reg.register(card)
    return reg


def _make_card(name: str, capabilities: list[str], tier: ModelTier = ModelTier.WARM) -> ModelCard:
    return ModelCard(
        name=name,
        display_name=name,
        backend=ModelBackend.OLLAMA,
        vram_mb=4096,
        ram_mb=4096,
        capabilities=capabilities,
        default_tier=tier,
        current_tier=tier,
    )


class TestContractValidation:
    def test_satisfied_when_model_available(self):
        registry = _make_registry(
            _make_card("qwen3:14b", ["reasoning"]),
        )
        manifest = ModuleManifest(
            id="test", name="Test", description="Test module",
            models=[
                ModelRequirement(capability="reasoning", preferred_model="qwen3:14b"),
            ],
        )
        report = validate_module_contract(manifest, registry)
        assert report.satisfied
        assert len(report.missing) == 0
        assert report.results[0].model_name == "qwen3:14b"

    def test_unsatisfied_when_model_missing(self):
        registry = _make_registry()  # Empty
        manifest = ModuleManifest(
            id="test", name="Test", description="Test module",
            models=[
                ModelRequirement(capability="vision", required=True),
            ],
        )
        report = validate_module_contract(manifest, registry)
        assert not report.satisfied
        assert "vision" in report.missing

    def test_optional_missing_doesnt_block(self):
        registry = _make_registry()  # Empty
        manifest = ModuleManifest(
            id="test", name="Test", description="Test module",
            models=[
                ModelRequirement(capability="embedding", required=False),
            ],
        )
        report = validate_module_contract(manifest, registry)
        assert report.satisfied  # Optional missing is OK
        assert "embedding" in report.optional_missing

    def test_fallback_to_capability_match(self):
        # No preferred model, but another model has the capability
        registry = _make_registry(
            _make_card("some-vision-model", ["vision"]),
        )
        manifest = ModuleManifest(
            id="test", name="Test", description="Test module",
            models=[
                ModelRequirement(capability="vision", preferred_model="qwen2.5vl:7b"),
            ],
        )
        report = validate_module_contract(manifest, registry)
        assert report.satisfied
        assert report.results[0].model_name == "some-vision-model"

    def test_tier_check_fails(self):
        registry = _make_registry(
            _make_card("qwen3:14b", ["reasoning"], tier=ModelTier.COLD),
        )
        manifest = ModuleManifest(
            id="test", name="Test", description="Test module",
            models=[
                ModelRequirement(
                    capability="reasoning",
                    min_tier="warm",  # Requires warm, but model is cold
                ),
            ],
        )
        report = validate_module_contract(manifest, registry)
        assert not report.satisfied  # Cold doesn't meet warm requirement

    def test_tier_check_passes_when_better(self):
        registry = _make_registry(
            _make_card("qwen3:14b", ["reasoning"], tier=ModelTier.RESIDENT),
        )
        manifest = ModuleManifest(
            id="test", name="Test", description="Test module",
            models=[
                ModelRequirement(capability="reasoning", min_tier="warm"),
            ],
        )
        report = validate_module_contract(manifest, registry)
        assert report.satisfied  # Resident is better than warm

    def test_multiple_requirements(self):
        registry = _make_registry(
            _make_card("qwen3:14b", ["reasoning"]),
            _make_card("qwen2.5vl:7b", ["vision"]),
        )
        manifest = ModuleManifest(
            id="test", name="Test", description="Test module",
            models=[
                ModelRequirement(capability="reasoning", required=True),
                ModelRequirement(capability="vision", required=True),
                ModelRequirement(capability="embedding", required=False),
            ],
        )
        report = validate_module_contract(manifest, registry)
        assert report.satisfied
        assert len(report.optional_missing) == 1

    def test_validate_contracts_skips_empty(self):
        registry = _make_registry()
        manifests = [
            ModuleManifest(id="no-models", name="NoModels", description="No model needs"),
            ModuleManifest(
                id="has-models", name="HasModels", description="Needs a model",
                models=[ModelRequirement(capability="reasoning")],
            ),
        ]
        reports = validate_contracts(registry, manifests=manifests)
        assert len(reports) == 1  # Only the one with models
        assert reports[0].module_id == "has-models"

    def test_validate_contracts_discovers_from_registry(self):
        registry = _make_registry(
            _make_card("qwen3:14b", ["reasoning", "agent", "gate"]),
            _make_card("qwen2.5vl:7b", ["vision", "escalation", "desktop"]),
        )
        # Uses real manifests via discover
        reports = validate_contracts(registry)
        assert len(reports) >= 4  # core, desktop, gate, research, agent, router all have models
        module_ids = {r.module_id for r in reports}
        assert "desktop" in module_ids
        assert "gate" in module_ids


class TestContractReport:
    def test_report_properties(self):
        report = ContractReport(module_id="test", module_name="Test")
        report.results = [
            RequirementResult(
                requirement=ModelRequirement(capability="reasoning"),
                available=True,
                model_name="qwen3:14b",
            ),
            RequirementResult(
                requirement=ModelRequirement(capability="vision", required=False),
                available=False,
            ),
        ]
        assert report.satisfied
        assert len(report.missing) == 0
        assert report.optional_missing == ["vision"]
