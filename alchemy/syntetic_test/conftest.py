"""Root conftest — shared config for the synthetic testing environment.

Services:
  Alchemy  = http://localhost:8000
  Voice    = http://localhost:8100
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ALCHEMY_URL = os.getenv("ALCHEMY_URL", "http://localhost:8000")
VOICE_URL = os.getenv("VOICE_URL", "http://localhost:8100")

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def alchemy_url() -> str:
    return ALCHEMY_URL


@pytest.fixture(scope="session")
def voice_url() -> str:
    return VOICE_URL


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR
