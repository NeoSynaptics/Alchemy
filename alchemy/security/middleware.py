"""Bearer token authentication middleware."""

from __future__ import annotations

import logging
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# Paths that skip authentication even when security is enabled.
_PUBLIC_PATHS = frozenset({"/health", "/docs", "/openapi.json", "/redoc"})


def create_auth_middleware(
    token: str,
    enabled: bool,
) -> Callable:
    """Return an ASGI middleware callable for bearer token validation.

    Parameters
    ----------
    token:
        Expected bearer token value.
    enabled:
        If False, the middleware is a no-op (all requests pass through).
    """

    async def auth_middleware(request: Request, call_next: Callable) -> Response:
        if not enabled:
            return await call_next(request)

        # Skip auth for public paths
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or malformed Authorization header"},
            )

        provided_token = auth_header[7:]  # len("Bearer ") == 7
        if provided_token != token:
            logger.warning("Rejected request with invalid token to %s", request.url.path)
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid API token"},
            )

        return await call_next(request)

    return auth_middleware
