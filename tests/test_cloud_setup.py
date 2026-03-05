"""Tests for Cloud AI Bridge — provider registry + setup module."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alchemy.cloud.providers import CloudProvider, get_provider, list_providers
from alchemy.cloud.setup import CloudSetup


class TestProviderRegistry:
    def test_list_providers_returns_all(self):
        providers = list_providers()
        assert len(providers) >= 3
        ids = [p.id for p in providers]
        assert "claude" in ids
        assert "openai" in ids
        assert "gemini" in ids

    def test_default_provider_is_claude(self):
        providers = list_providers()
        assert providers[0].id == "claude"
        assert providers[0].default is True

    def test_get_provider_by_id(self):
        p = get_provider("claude")
        assert p is not None
        assert p.name == "Claude (Anthropic)"
        assert p.env_key == "ANTHROPIC_API_KEY"

    def test_get_unknown_provider(self):
        assert get_provider("nonexistent") is None

    def test_claude_has_vscode_extension(self):
        p = get_provider("claude")
        assert p.vscode_extension == "anthropic.claude-code"

    def test_provider_has_required_fields(self):
        for p in list_providers():
            assert p.id
            assert p.name
            assert p.env_key
            assert p.setup_instructions


class TestCloudSetup:
    def test_store_key(self, tmp_path, monkeypatch):
        monkeypatch.setattr("alchemy.cloud.setup._CLOUD_CONFIG_DIR", tmp_path)
        setup = CloudSetup()
        result = setup.store_key("claude", "sk-ant-test123")
        assert result.success is True
        assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-test123"
        assert (tmp_path / "claude.env").exists()
        # Cleanup
        del os.environ["ANTHROPIC_API_KEY"]

    def test_store_key_empty_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr("alchemy.cloud.setup._CLOUD_CONFIG_DIR", tmp_path)
        setup = CloudSetup()
        result = setup.store_key("claude", "")
        assert result.success is False

    def test_store_key_unknown_provider(self, tmp_path, monkeypatch):
        monkeypatch.setattr("alchemy.cloud.setup._CLOUD_CONFIG_DIR", tmp_path)
        setup = CloudSetup()
        result = setup.store_key("fakeprovider", "key123")
        assert result.success is False

    def test_load_key(self, tmp_path, monkeypatch):
        monkeypatch.setattr("alchemy.cloud.setup._CLOUD_CONFIG_DIR", tmp_path)
        (tmp_path / "claude.env").write_text("ANTHROPIC_API_KEY=sk-test\n")
        setup = CloudSetup()
        key = setup._load_key("claude")
        assert key == "sk-test"

    def test_load_key_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("alchemy.cloud.setup._CLOUD_CONFIG_DIR", tmp_path)
        setup = CloudSetup()
        assert setup._load_key("claude") is None

    def test_get_status_no_key(self, tmp_path, monkeypatch):
        monkeypatch.setattr("alchemy.cloud.setup._CLOUD_CONFIG_DIR", tmp_path)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        setup = CloudSetup()
        status = setup.get_status("claude")
        assert status["has_key"] is False
        assert status["ready"] is False

    def test_get_status_with_key(self, tmp_path, monkeypatch):
        monkeypatch.setattr("alchemy.cloud.setup._CLOUD_CONFIG_DIR", tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        setup = CloudSetup()
        with patch.object(setup, "_check_extension", return_value=True):
            status = setup.get_status("claude")
        assert status["has_key"] is True

    def test_get_status_unknown(self, tmp_path, monkeypatch):
        monkeypatch.setattr("alchemy.cloud.setup._CLOUD_CONFIG_DIR", tmp_path)
        setup = CloudSetup()
        status = setup.get_status("fake")
        assert "error" in status

    def test_load_all_keys(self, tmp_path, monkeypatch):
        monkeypatch.setattr("alchemy.cloud.setup._CLOUD_CONFIG_DIR", tmp_path)
        (tmp_path / "claude.env").write_text("ANTHROPIC_API_KEY=sk-1\n")
        (tmp_path / "openai.env").write_text("OPENAI_API_KEY=sk-2\n")
        setup = CloudSetup()
        count = setup.load_all_keys()
        assert count == 2
        assert os.environ.get("ANTHROPIC_API_KEY") == "sk-1"
        assert os.environ.get("OPENAI_API_KEY") == "sk-2"
        # Cleanup
        del os.environ["ANTHROPIC_API_KEY"]
        del os.environ["OPENAI_API_KEY"]

    @patch("subprocess.run")
    def test_install_extension(self, mock_run, tmp_path, monkeypatch):
        monkeypatch.setattr("alchemy.cloud.setup._CLOUD_CONFIG_DIR", tmp_path)
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        setup = CloudSetup()
        result = setup.install_extension("claude")
        assert result.success is True
        mock_run.assert_called_once()

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_install_extension_no_vscode(self, mock_run, tmp_path, monkeypatch):
        monkeypatch.setattr("alchemy.cloud.setup._CLOUD_CONFIG_DIR", tmp_path)
        setup = CloudSetup()
        result = setup.install_extension("claude")
        assert result.success is False
        assert "not found" in result.message

    def test_install_extension_not_needed(self, tmp_path, monkeypatch):
        monkeypatch.setattr("alchemy.cloud.setup._CLOUD_CONFIG_DIR", tmp_path)
        setup = CloudSetup()
        result = setup.install_extension("gemini")  # No extension
        assert result.success is True

    @pytest.mark.asyncio
    async def test_validate_success(self, tmp_path, monkeypatch):
        monkeypatch.setattr("alchemy.cloud.setup._CLOUD_CONFIG_DIR", tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        setup = CloudSetup()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(return_value=MagicMock(
                get=AsyncMock(return_value=mock_resp)
            ))
            mock_client.return_value.__aexit__ = AsyncMock()
            result = await setup.validate("claude")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_validate_no_key(self, tmp_path, monkeypatch):
        monkeypatch.setattr("alchemy.cloud.setup._CLOUD_CONFIG_DIR", tmp_path)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        setup = CloudSetup()
        result = await setup.validate("claude")
        assert result.success is False
        assert "No API key" in result.message

    @pytest.mark.asyncio
    async def test_validate_no_endpoint(self, tmp_path, monkeypatch):
        monkeypatch.setattr("alchemy.cloud.setup._CLOUD_CONFIG_DIR", tmp_path)
        monkeypatch.setenv("GOOGLE_API_KEY", "test")
        setup = CloudSetup()
        result = await setup.validate("gemini")
        assert result.success is True  # No validation URL = assume OK
