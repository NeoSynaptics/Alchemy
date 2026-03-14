"""NEO-N API routes — file upload and inbox management.

Endpoints:
  POST /v1/neo-n/upload     — Multipart file upload from paired device
  GET  /v1/neo-n/pending     — List pending (unprocessed) files
  GET  /v1/neo-n/stats       — Inbox statistics
  POST /v1/neo-n/processed   — Mark file as processed (called by BaratzaMemory)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile

from alchemy.neo_n.receiver import FileReceiver

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/neo-n", tags=["neo-n"])

# Set by server.py lifespan
_receiver: FileReceiver | None = None
_pairing_manager: Any = None  # AlchemyConnect PairingManager


def init(receiver: FileReceiver, pairing_manager: Any = None) -> None:
    """Wire the receiver and pairing manager from server lifespan."""
    global _receiver, _pairing_manager
    _receiver = receiver
    _pairing_manager = pairing_manager


def _get_receiver() -> FileReceiver:
    if _receiver is None:
        raise HTTPException(503, "NEO-N not initialized")
    return _receiver


def _verify_device(authorization: str | None) -> tuple[str, str]:
    """Verify device token via AlchemyConnect's PairingManager.

    Returns (device_id, device_name) or raises 401.
    """
    if not authorization:
        raise HTTPException(401, "Authorization header required")

    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(401, "Bearer token required")

    if _pairing_manager is None:
        # No pairing manager — accept any non-empty token (dev mode)
        logger.warning("NEO-N: No pairing manager, accepting token in dev mode")
        return ("dev_device", "Dev Device")

    device = _pairing_manager.verify_token(token)
    if not device:
        raise HTTPException(401, "Invalid or revoked device token")

    return (device.device_id, device.device_name)


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    metadata: str = Form("{}"),
    authorization: str | None = Header(None),
):
    """Upload a file from a paired device.

    Accepts multipart form with:
      - file: the file to upload
      - metadata: JSON string with optional tags, notes, source_app
    """
    receiver = _get_receiver()
    device_id, device_name = _verify_device(authorization)

    # Parse metadata
    try:
        meta = json.loads(metadata) if metadata else {}
    except json.JSONDecodeError:
        meta = {}

    # Read file data
    file_data = await file.read()
    original_filename = file.filename or "unknown"
    content_type = file.content_type or "application/octet-stream"

    result = await receiver.save(
        file_data=file_data,
        original_filename=original_filename,
        content_type=content_type,
        device_id=device_id,
        device_name=device_name,
        metadata=meta,
    )

    if not result.success:
        raise HTTPException(400, result.error)

    return {
        "status": "received",
        "filename": result.filename,
        "size_bytes": result.size_bytes,
    }


@router.get("/pending")
async def list_pending():
    """List files waiting to be processed by BaratzaMemory."""
    receiver = _get_receiver()
    return {"files": receiver.list_pending()}


@router.get("/stats")
async def inbox_stats():
    """Inbox statistics — pending/processed counts and sizes."""
    receiver = _get_receiver()
    return receiver.stats()


@router.post("/processed")
async def mark_processed(filename: str):
    """Mark a file as processed (moves to processed/ subfolder).

    Called by BaratzaMemory after successful ingest.
    """
    receiver = _get_receiver()
    ok = receiver.mark_processed(filename)
    if not ok:
        raise HTTPException(404, f"File not found: {filename}")
    return {"status": "processed", "filename": filename}
