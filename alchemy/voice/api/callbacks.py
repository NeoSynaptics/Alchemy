"""Callback endpoints — Alchemy calls these to communicate with AlchemyVoice.

Approval requests block until user decides (or timeout). Notifications
and task updates are dispatched to the tray and acknowledged immediately.
If tray is disabled, approvals auto-approve as fallback.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Request

from alchemy.voice.schemas import (
    ApprovalRequest,
    ApprovalRequestAck,
    NotifyAck,
    NotifyRequest,
    TaskUpdateAck,
    TaskUpdateRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/callbacks", tags=["callbacks"])


@router.post("/approval", response_model=ApprovalRequestAck)
async def receive_approval_request(
    req: ApprovalRequest, request: Request
) -> ApprovalRequestAck:
    """Alchemy asks: user needs to approve this action before it executes.

    If tray is active, shows an approval dialog and waits for user response.
    If tray is disabled, auto-approves as fallback.
    """
    from alchemy.voice.constitution.engine import ConstitutionEngine
    from alchemy.voice.tray.events import TrayEvent, TrayMessage

    event_bus = getattr(request.app.state, "tray_event_bus", None)
    alchemy_client = getattr(request.app.state, "alchemy_client", None)
    constitution: ConstitutionEngine | None = getattr(
        request.app.state, "constitution", None
    )

    # Run constitutional rules — may escalate tier
    if constitution is not None:
        enforced = constitution.enforce(req.action, req.goal)
        if enforced.tier != req.action.tier:
            logger.info(
                "Constitution escalated action tier %s → %s for task %s",
                req.action.tier.value,
                enforced.tier.value,
                req.task_id,
            )
            req.action = enforced

    if event_bus is None:
        # Tray disabled — auto-deny for safety (fail-closed)
        logger.warning("Tray disabled: auto-denying task %s (no approval UI available)", req.task_id)
        if alchemy_client:
            try:
                await alchemy_client.deny_task(req.task_id, decided_by="auto-deny-no-tray")
            except Exception:
                logger.exception("Failed to forward auto-denial to Alchemy")
        return ApprovalRequestAck(received=True, task_id=req.task_id)

    # Create Future for GUI thread to resolve
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    msg = TrayMessage(
        event=TrayEvent.APPROVAL_REQUEST,
        data=req.model_dump(mode="json"),
        response_future=future,
    )
    event_bus.push(msg)

    # Wait for user decision (or timeout)
    try:
        approved = await asyncio.wait_for(future, timeout=req.timeout_seconds)
    except asyncio.TimeoutError:
        approved = False
        logger.info("Approval timed out for task %s — auto-denied", req.task_id)

    # Forward decision to Alchemy
    if alchemy_client:
        try:
            if approved:
                await alchemy_client.approve_task(req.task_id, decided_by="user")
            else:
                await alchemy_client.deny_task(req.task_id, decided_by="user")
        except Exception:
            logger.exception("Failed to forward approval decision to Alchemy")

    return ApprovalRequestAck(received=True, task_id=req.task_id)


@router.post("/notify", response_model=NotifyAck)
async def receive_notification(req: NotifyRequest, request: Request) -> NotifyAck:
    """Alchemy notifies: a NOTIFY-tier action was executed."""
    from alchemy.voice.tray.events import TrayEvent, TrayMessage

    event_bus = getattr(request.app.state, "tray_event_bus", None)

    if event_bus is not None:
        msg = TrayMessage(
            event=TrayEvent.NOTIFICATION,
            data=req.model_dump(mode="json"),
        )
        event_bus.push(msg)

    return NotifyAck(received=True)


@router.post("/task-update", response_model=TaskUpdateAck)
async def receive_task_update(
    req: TaskUpdateRequest, request: Request
) -> TaskUpdateAck:
    """Alchemy reports: task status changed (running, completed, failed)."""
    from alchemy.voice.tray.events import TrayEvent, TrayMessage

    event_bus = getattr(request.app.state, "tray_event_bus", None)

    if event_bus is not None:
        msg = TrayMessage(
            event=TrayEvent.TASK_UPDATE,
            data=req.model_dump(mode="json"),
        )
        event_bus.push(msg)

    return TaskUpdateAck(received=True)
