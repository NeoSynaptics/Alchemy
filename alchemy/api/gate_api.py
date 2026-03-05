"""Gate API — Claude Code tool call reviewer endpoint.

POST /gate/review — receives tool_name + args, returns accept/deny/other.
Matches the contract expected by AlchemyVS gateClient.ts.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["gate"])


class GateReviewRequest(BaseModel):
    tool_name: str
    args: dict[str, str] = {}
    project_context: dict[str, str] = {}


class GateReviewResponse(BaseModel):
    action: str  # "accept" | "deny" | "other"
    reason: str
    tier: str  # "static" | "ask_ollama"
    latency_ms: float = 0.0
    model: str = ""


@router.post("/review")
async def review_tool_call(
    req: GateReviewRequest,
    request: Request,
) -> GateReviewResponse:
    """Review a Claude Code tool call for auto-approval."""
    gate_reviewer = getattr(request.app.state, "gate_reviewer", None)

    if gate_reviewer is None:
        logger.warning("Gate reviewer not initialized, fail-open accept")
        return GateReviewResponse(
            action="accept",
            reason="gate not initialized",
            tier="static",
        )

    result = await gate_reviewer.review(
        tool_name=req.tool_name,
        args=req.args,
        project_context=req.project_context,
    )

    return GateReviewResponse(
        action=result.action,
        reason=result.reason,
        tier=result.tier,
        latency_ms=result.latency_ms,
        model=result.model,
    )
