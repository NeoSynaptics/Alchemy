"""Shadow Desktop API — start, stop, health, screenshot.

AlchemyVoice calls these to manage the WSL2 shadow desktop.
Uses ShadowDesktopController for real WSL2 operations,
falls back to mock responses if WSL2 is unavailable.
"""

from __future__ import annotations

import logging
from io import BytesIO

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from PIL import Image

from alchemy.schemas import (
    ShadowHealthResponse,
    ShadowStartRequest,
    ShadowStartResponse,
    ShadowStatus,
    ShadowStopResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/shadow", tags=["shadow"])


def _get_controller(request: Request):
    """Get the ShadowDesktopController from app state (set in server lifespan)."""
    return getattr(request.app.state, "shadow_controller", None)


@router.post("/start", response_model=ShadowStartResponse)
async def start_shadow(request: Request, req: ShadowStartRequest | None = None) -> ShadowStartResponse:
    """Start the shadow desktop (Xvfb + Fluxbox + x11vnc + noVNC)."""
    controller = _get_controller(request)
    if controller:
        return await controller.start(req)

    # Fallback: mock response when WSL2 not available
    req = req or ShadowStartRequest()
    return ShadowStartResponse(
        status=ShadowStatus.ERROR,
        display=f":{req.display_num}",
        vnc_url="localhost:5900",
        novnc_url="",
    )


@router.post("/stop", response_model=ShadowStopResponse)
async def stop_shadow(request: Request) -> ShadowStopResponse:
    """Stop the shadow desktop."""
    controller = _get_controller(request)
    if controller:
        return await controller.stop()

    return ShadowStopResponse(status=ShadowStatus.STOPPED)


@router.get("/health", response_model=ShadowHealthResponse)
async def shadow_health(request: Request) -> ShadowHealthResponse:
    """Check if the shadow desktop services are running."""
    controller = _get_controller(request)
    if controller:
        return await controller.health()

    return ShadowHealthResponse(status=ShadowStatus.STOPPED)


@router.get("/screenshot")
async def screenshot(request: Request) -> Response:
    """Capture a screenshot from the shadow desktop. Returns PNG bytes."""
    controller = _get_controller(request)
    if controller:
        try:
            png_bytes = await controller.screenshot()
            return Response(content=png_bytes, media_type="image/png")
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

    # Fallback: gray placeholder
    img = Image.new("RGB", (1920, 1080), color=(64, 64, 64))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
