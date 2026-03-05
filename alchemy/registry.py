"""Module registry — discovers all Alchemy modules with manifests.

Scans alchemy/*/manifest.py for MANIFEST objects. Used by setup wizards,
settings pages, and the GET /v1/modules discovery API.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

from alchemy.manifest import ModuleManifest

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, ModuleManifest] = {}


def discover() -> dict[str, ModuleManifest]:
    """Scan alchemy/*/manifest.py and collect MANIFEST objects."""
    if _REGISTRY:
        return _REGISTRY

    alchemy_dir = Path(__file__).parent
    for child in sorted(alchemy_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        manifest_file = child / "manifest.py"
        if not manifest_file.exists():
            continue
        try:
            mod = importlib.import_module(f"alchemy.{child.name}.manifest")
            manifest = getattr(mod, "MANIFEST", None)
            if isinstance(manifest, ModuleManifest):
                _REGISTRY[manifest.id] = manifest
        except Exception as e:
            logger.warning("Failed to load manifest for %s: %s", child.name, e)

    return _REGISTRY


def get(module_id: str) -> ModuleManifest | None:
    """Get a module manifest by ID."""
    if not _REGISTRY:
        discover()
    return _REGISTRY.get(module_id)


def all_modules() -> list[ModuleManifest]:
    """List all discovered module manifests."""
    if not _REGISTRY:
        discover()
    return list(_REGISTRY.values())


def reset() -> None:
    """Clear the registry (for testing)."""
    _REGISTRY.clear()
