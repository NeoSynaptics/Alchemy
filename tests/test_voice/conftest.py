"""AlchemyVoice test configuration."""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from alchemy.server import app


@pytest.fixture
def alchemy_url():
    """Default Alchemy API URL."""
    return "http://localhost:8000"


@pytest_asyncio.fixture
async def async_client():
    """Async HTTP client for testing voice endpoints via Alchemy server."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
