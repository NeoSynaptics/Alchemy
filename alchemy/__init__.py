"""Alchemy — Local-first LLM core engine."""

from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("alchemy")
except Exception:
    __version__ = "0.4.0"
