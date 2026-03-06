"""Fixtures for tray tests — mock PyQt6 to avoid display requirement."""

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def mock_pyqt6():
    """Mock PyQt6 modules so tray tests run without a display server."""
    modules_to_mock = [
        "PyQt6",
        "PyQt6.QtCore",
        "PyQt6.QtGui",
        "PyQt6.QtWidgets",
        "PyQt6.QtWebEngineWidgets",
    ]
    mocks = {}
    originals = {}

    for mod_name in modules_to_mock:
        originals[mod_name] = sys.modules.get(mod_name)
        mock = MagicMock()
        sys.modules[mod_name] = mock
        mocks[mod_name] = mock

    yield mocks

    # Restore originals
    for mod_name in modules_to_mock:
        if originals[mod_name] is None:
            sys.modules.pop(mod_name, None)
        else:
            sys.modules[mod_name] = originals[mod_name]
