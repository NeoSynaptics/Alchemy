"""Tests for module registry and manifest discovery."""

from __future__ import annotations

from alchemy.manifest import ModuleManifest
from alchemy.registry import all_modules, discover, get, reset


class TestModuleRegistry:
    def setup_method(self):
        reset()

    def test_discover_finds_modules(self):
        manifests = discover()
        assert len(manifests) >= 5
        assert "gate" in manifests
        assert "desktop" in manifests
        assert "cloud" in manifests

    def test_get_existing_module(self):
        m = get("gate")
        assert m is not None
        assert m.id == "gate"
        assert m.name == "Gate Reviewer"

    def test_get_missing_module(self):
        assert get("nonexistent") is None

    def test_all_modules_returns_list(self):
        modules = all_modules()
        assert isinstance(modules, list)
        assert all(isinstance(m, ModuleManifest) for m in modules)

    def test_no_duplicate_ids(self):
        modules = all_modules()
        ids = [m.id for m in modules]
        assert len(ids) == len(set(ids)), f"Duplicate IDs found: {ids}"

    def test_reset_clears_registry(self):
        discover()
        reset()
        # After reset, next call should re-discover
        modules = all_modules()
        assert len(modules) >= 5


class TestManifestFields:
    """Every manifest must have required fields and valid tier."""

    def setup_method(self):
        reset()

    def test_all_manifests_have_required_fields(self):
        for m in all_modules():
            assert m.id, f"Manifest missing id"
            assert m.name, f"Manifest {m.id} missing name"
            assert m.description, f"Manifest {m.id} missing description"
            assert m.tier in ("core", "infra", "app"), (
                f"Manifest {m.id} has invalid tier: {m.tier}"
            )

    def test_core_tier_modules(self):
        core_mods = [m for m in all_modules() if m.tier == "core"]
        core_ids = {m.id for m in core_mods}
        assert "core" in core_ids
        assert "desktop" in core_ids
        assert "gpu" in core_ids
        assert "cloud" in core_ids

    def test_infra_tier_modules(self):
        infra_mods = [m for m in all_modules() if m.tier == "infra"]
        infra_ids = {m.id for m in infra_mods}
        assert "adapters" in infra_ids
        assert "shadow" in infra_ids

    def test_enabled_key_exists_where_declared(self):
        for m in all_modules():
            if m.enabled_key:
                # Verify the enabled_key follows convention
                assert m.enabled_key.endswith("_enabled"), (
                    f"Manifest {m.id} enabled_key should end with '_enabled': {m.enabled_key}"
                )
