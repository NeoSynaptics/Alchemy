"""Alchemy test configuration."""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from alchemy.server import app


@pytest.fixture
def display_num():
    """Default display number for shadow desktop tests."""
    return 99


@pytest.fixture
def vnc_port():
    """Default VNC port."""
    return 5900


@pytest.fixture
def novnc_port():
    """Default noVNC port."""
    return 6080


@pytest_asyncio.fixture
async def async_client():
    """Async HTTP client for testing Alchemy endpoints."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
