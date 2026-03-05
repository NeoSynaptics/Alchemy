"""Cloud AI setup — callable by a setup wizard, CLI, or API.

Handles:
1. Store API key securely
2. Install VS Code extension if needed
3. Validate credentials work
4. Report status

Designed so a GUI wizard just calls these methods in sequence.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import httpx

from alchemy.cloud.providers import CloudProvider, get_provider

logger = logging.getLogger(__name__)

# Where we store cloud config (inside Alchemy's data dir)
_CLOUD_CONFIG_DIR = Path.home() / ".alchemy" / "cloud"


@dataclass
class SetupResult:
    """Result of a setup step."""
    success: bool
    message: str
    provider_id: str


class CloudSetup:
    """Manages cloud AI provider setup.

    Usage (from a setup wizard):
        setup = CloudSetup()
        providers = setup.list_providers()       # Show to user
        result = setup.store_key("claude", key)  # User entered key
        result = setup.install_extension("claude")  # Install VS Code ext
        result = await setup.validate("claude")  # Check it works
        status = setup.get_status("claude")      # Current state
    """

    def __init__(self):
        _CLOUD_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    def store_key(self, provider_id: str, api_key: str) -> SetupResult:
        """Store an API key for a provider.

        Writes to ~/.alchemy/cloud/{provider_id}.env
        The key is also set in the current process environment.
        """
        provider = get_provider(provider_id)
        if not provider:
            return SetupResult(False, f"Unknown provider: {provider_id}", provider_id)

        if not api_key or not api_key.strip():
            return SetupResult(False, "API key cannot be empty", provider_id)

        # Store in env file
        env_file = _CLOUD_CONFIG_DIR / f"{provider_id}.env"
        env_file.write_text(f"{provider.env_key}={api_key.strip()}\n")

        # Set in current process
        os.environ[provider.env_key] = api_key.strip()

        logger.info("Stored API key for %s", provider.name)
        return SetupResult(True, f"API key stored for {provider.name}", provider_id)

    def install_extension(self, provider_id: str) -> SetupResult:
        """Install the VS Code extension for a provider."""
        provider = get_provider(provider_id)
        if not provider:
            return SetupResult(False, f"Unknown provider: {provider_id}", provider_id)

        if not provider.vscode_extension:
            return SetupResult(True, f"No extension needed for {provider.name}", provider_id)

        try:
            result = subprocess.run(
                ["code", "--install-extension", provider.vscode_extension, "--force"],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                logger.info("Installed VS Code extension: %s", provider.vscode_extension)
                return SetupResult(True, f"Installed {provider.vscode_extension}", provider_id)
            else:
                return SetupResult(False, f"Install failed: {result.stderr[:200]}", provider_id)
        except FileNotFoundError:
            return SetupResult(False, "VS Code CLI (code) not found in PATH", provider_id)
        except subprocess.TimeoutExpired:
            return SetupResult(False, "Extension install timed out", provider_id)

    async def validate(self, provider_id: str) -> SetupResult:
        """Validate that stored credentials work by pinging the provider API."""
        provider = get_provider(provider_id)
        if not provider:
            return SetupResult(False, f"Unknown provider: {provider_id}", provider_id)

        api_key = os.environ.get(provider.env_key) or self._load_key(provider_id)
        if not api_key:
            return SetupResult(False, f"No API key found for {provider.name}", provider_id)

        if not provider.validate_url:
            return SetupResult(True, f"No validation endpoint for {provider.name} (key stored)", provider_id)

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                headers = self._auth_headers(provider, api_key)
                resp = await client.get(provider.validate_url, headers=headers)
                # 200 or 401-with-body both mean "server reached" — 401 means bad key
                if resp.status_code in (200, 201):
                    return SetupResult(True, f"{provider.name} credentials valid", provider_id)
                elif resp.status_code == 401:
                    return SetupResult(False, f"Invalid API key for {provider.name}", provider_id)
                else:
                    return SetupResult(True, f"{provider.name} reachable (HTTP {resp.status_code})", provider_id)
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            return SetupResult(False, f"Cannot reach {provider.name}: {e}", provider_id)

    def get_status(self, provider_id: str) -> dict:
        """Get current setup status for a provider.

        Returns dict with: provider_id, name, has_key, has_extension, ready
        """
        provider = get_provider(provider_id)
        if not provider:
            return {"provider_id": provider_id, "error": "unknown provider"}

        has_key = bool(os.environ.get(provider.env_key) or self._load_key(provider_id))
        has_extension = self._check_extension(provider) if provider.vscode_extension else True

        return {
            "provider_id": provider_id,
            "name": provider.name,
            "has_key": has_key,
            "has_extension": has_extension,
            "ready": has_key and has_extension,
        }

    def load_all_keys(self) -> int:
        """Load all stored API keys into the environment. Returns count loaded."""
        count = 0
        for env_file in _CLOUD_CONFIG_DIR.glob("*.env"):
            for line in env_file.read_text().strip().splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
                    count += 1
        return count

    def _load_key(self, provider_id: str) -> str | None:
        """Load API key from stored env file."""
        provider = get_provider(provider_id)
        if not provider:
            return None
        env_file = _CLOUD_CONFIG_DIR / f"{provider_id}.env"
        if not env_file.exists():
            return None
        for line in env_file.read_text().strip().splitlines():
            if line.startswith(provider.env_key):
                return line.split("=", 1)[1].strip()
        return None

    def _check_extension(self, provider: CloudProvider) -> bool:
        """Check if a VS Code extension is installed."""
        if not provider.vscode_extension:
            return True
        try:
            result = subprocess.run(
                ["code", "--list-extensions"],
                capture_output=True, text=True, timeout=10,
            )
            return provider.vscode_extension in result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def _auth_headers(provider: CloudProvider, api_key: str) -> dict:
        """Build auth headers for a provider."""
        if provider.id == "claude":
            return {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
        elif provider.id == "openai":
            return {"Authorization": f"Bearer {api_key}"}
        else:
            return {"Authorization": f"Bearer {api_key}"}
