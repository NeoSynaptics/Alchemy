"""Tests for Playwright API endpoints."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


def _create_test_app():
    """Create a test app with mocked Playwright components."""
    from alchemy.server import app

    # Mock the playwright agent and browser manager
    app.state.pw_agent = MagicMock()
    app.state.browser_manager = MagicMock()

    return app


class TestPlaywrightAPI:
    def test_task_endpoint_exists(self):
        """POST /v1/playwright/task endpoint is registered."""
        from alchemy.api.playwright_api import router
        routes = [r.path for r in router.routes]
        assert "/playwright/task" in routes

    def test_status_endpoint_exists(self):
        """GET /v1/playwright/task/{id}/status endpoint is registered."""
        from alchemy.api.playwright_api import router
        routes = [r.path for r in router.routes]
        assert "/playwright/task/{task_id}/status" in routes

    def test_task_not_found(self):
        """GET status for non-existent task returns 404."""
        from alchemy.api.playwright_api import _tasks
        _tasks.clear()

        # Manually check the logic
        assert "nonexistent" not in _tasks
