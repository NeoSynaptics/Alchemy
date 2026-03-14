"""Tests for NEO-N manifest and module discovery."""

from alchemy.neo_n.manifest import MANIFEST


class TestNeoNManifest:
    def test_manifest_id(self):
        assert MANIFEST.id == "neo_n"

    def test_manifest_tier(self):
        assert MANIFEST.tier == "app"

    def test_manifest_no_models(self):
        assert MANIFEST.models == []

    def test_manifest_requires_connect(self):
        assert "connect" in MANIFEST.requires

    def test_manifest_api_prefix(self):
        assert MANIFEST.api_prefix == "/v1/neo-n"

    def test_discoverable(self):
        from alchemy.registry import discover
        registry = discover()
        assert "neo_n" in registry
        assert registry["neo_n"].name == "NEO-N"
