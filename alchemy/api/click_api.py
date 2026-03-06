"""AlchemyClick function call API — external entry point.

Exposes AlchemyClick, AlchemyFlow, and AlchemyBrowser as callable
functions. Only functions with EXTERNAL or BOTH visibility are served.

Endpoints:
  POST /v1/click/call           — invoke alchemy_click (auto-routes)
  POST /v1/click/flow           — invoke alchemy_flow directly
  POST /v1/click/browser        — invoke alchemy_browser directly
  GET  /v1/click/functions      — list available functions
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from alchemy.api.contract_guard import require_contract
from alchemy.click.functions import (
    dispatch_browser,
    dispatch_click,
    dispatch_flow,
    external_functions,
)
from alchemy.schemas import (
    ClickCallRequest,
    ClickCallResult,
    ClickTarget,
)

router = APIRouter(
    prefix="/v1/click",
    tags=["click"],
    dependencies=[Depends(require_contract("click"))],
)


@router.post("/call", response_model=ClickCallResult)
async def call_click(req: ClickCallRequest, request: Request) -> ClickCallResult:
    """Invoke alchemy_click — auto-routes to Flow or Browser."""
    return await dispatch_click(req, app_state=request.app.state)


@router.post("/flow", response_model=ClickCallResult)
async def call_flow(req: ClickCallRequest, request: Request) -> ClickCallResult:
    """Invoke alchemy_flow directly (vision + ghost cursor)."""
    req.target = ClickTarget.FLOW
    return await dispatch_flow(req, app_state=request.app.state)


@router.post("/browser", response_model=ClickCallResult)
async def call_browser(req: ClickCallRequest, request: Request) -> ClickCallResult:
    """Invoke alchemy_browser directly (Playwright + a11y)."""
    req.target = ClickTarget.BROWSER
    return await dispatch_browser(req, app_state=request.app.state)


@router.get("/functions")
async def list_functions():
    """List all externally-visible click functions."""
    return [
        {
            "name": fn.name,
            "description": fn.description,
            "target": fn.target.value,
            "params": fn.params,
        }
        for fn in external_functions()
    ]
