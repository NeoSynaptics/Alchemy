"""Security middleware tests — bearer token validation."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from alchemy.security.middleware import create_auth_middleware

SECRET = "test-secret-token-123"


def _make_app(token: str = SECRET, enabled: bool = True) -> FastAPI:
    """Create a minimal FastAPI app with the auth middleware."""
    app = FastAPI()

    _check = create_auth_middleware(token=token, enabled=enabled)

    @app.middleware("http")
    async def auth(request, call_next):
        return await _check(request, call_next)

    @app.get("/v1/apu/status")
    async def protected():
        return {"status": "ok"}

    @app.get("/health")
    async def health():
        return {"healthy": True}

    return app


class TestBearerAuth:
    """Bearer token authentication when security is enabled."""

    def setup_method(self):
        self.client = TestClient(_make_app(enabled=True))

    def test_valid_token_passes(self):
        resp = self.client.get(
            "/v1/apu/status",
            headers={"Authorization": f"Bearer {SECRET}"},
        )
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_missing_header_rejected(self):
        resp = self.client.get("/v1/apu/status")
        assert resp.status_code == 401

    def test_invalid_token_rejected(self):
        resp = self.client.get(
            "/v1/apu/status",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 401

    def test_malformed_header_rejected(self):
        resp = self.client.get(
            "/v1/apu/status",
            headers={"Authorization": "Basic abc123"},
        )
        assert resp.status_code == 401

    def test_health_skips_auth(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"healthy": True}


class TestSecurityDisabled:
    """When security is disabled, all requests pass through."""

    def setup_method(self):
        self.client = TestClient(_make_app(enabled=False))

    def test_no_token_passes(self):
        resp = self.client.get("/v1/apu/status")
        assert resp.status_code == 200

    def test_wrong_token_passes(self):
        resp = self.client.get(
            "/v1/apu/status",
            headers={"Authorization": "Bearer totally-wrong"},
        )
        assert resp.status_code == 200
