"""Shadow Desktop API — start, stop, health, screenshot.

NEO-TX calls these to manage the WSL2 shadow desktop.
All responses are stubs (Phase 0) — correct types, mock data.
"""

from __future__ import annotations

from io import BytesIO

from fastapi import APIRouter
from fastapi.responses import Response
from PIL import Image

from alchemy.schemas import (
    ShadowHealthResponse,
    ShadowStartRequest,
    ShadowStartResponse,
    ShadowStatus,
    ShadowStopResponse,
)

router = APIRouter(prefix="/shadow", tags=["shadow"])


@router.post("/start", response_model=ShadowStartResponse)
async def start_shadow(req: ShadowStartRequest | None = None) -> ShadowStartResponse:
    """Start the shadow desktop (Xvfb + Fluxbox + x11vnc + noVNC)."""
    req = req or ShadowStartRequest()
    return ShadowStartResponse(
        status=ShadowStatus.RUNNING,
        display=f":{req.display_num}",
        vnc_url="localhost:5900",
        novnc_url="http://localhost:6080/vnc.html?autoconnect=true",
    )


@router.post("/stop", response_model=ShadowStopResponse)
async def stop_shadow() -> ShadowStopResponse:
    """Stop the shadow desktop."""
    return ShadowStopResponse(status=ShadowStatus.STOPPED)


@router.get("/health", response_model=ShadowHealthResponse)
async def shadow_health() -> ShadowHealthResponse:
    """Check if the shadow desktop services are running."""
    return ShadowHealthResponse(
        status=ShadowStatus.STOPPED,
        xvfb_running=False,
        fluxbox_running=False,
        vnc_running=False,
        novnc_running=False,
    )


@router.get("/screenshot")
async def screenshot() -> Response:
    """Capture a screenshot from the shadow desktop. Returns PNG bytes."""
    img = Image.new("RGB", (1920, 1080), color=(64, 64, 64))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
