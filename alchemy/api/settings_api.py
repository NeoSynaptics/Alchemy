"""Settings read/write API — runtime access to config/settings.py."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from config.settings import settings

router = APIRouter(tags=["settings"])

# Nested group field names (excludes flat compat fields)
_NESTED_GROUPS = {
    "ollama", "server", "auth", "screenshot", "click", "router",
    "pw", "pw_escalation", "gui_actor", "desktop", "gate",
    "research", "word", "voice", "connect", "agents",
}


@router.get("/settings")
async def get_settings():
    """Return current settings grouped by module (nested groups only)."""
    full = settings.model_dump()
    return {k: full[k] for k in _NESTED_GROUPS if k in full}


@router.patch("/settings")
async def patch_settings(body: dict):
    """Update settings in-memory (runtime only, not persisted to disk).

    Accepts partial JSON keyed by group name:
        {"pw": {"temperature": 0.2}, "voice": {"tts_engine": "kokoro"}}
    """
    errors = []
    updated = []

    for group_name, patch in body.items():
        if group_name not in _NESTED_GROUPS:
            errors.append(f"Unknown settings group: {group_name!r}")
            continue

        if not isinstance(patch, dict):
            errors.append(f"{group_name}: value must be a JSON object")
            continue

        group_obj = getattr(settings, group_name, None)
        if group_obj is None:
            errors.append(f"{group_name}: group not found on settings")
            continue

        # Validate by constructing a new instance with merged values
        current = group_obj.model_dump()
        merged = {**current, **patch}
        try:
            validated = group_obj.__class__(**merged)
        except ValidationError as e:
            errors.append(f"{group_name}: {e}")
            continue

        # Apply validated values to the live settings object
        for field_name, value in validated.model_dump().items():
            if field_name in patch:
                setattr(group_obj, field_name, value)
                updated.append(f"{group_name}.{field_name}")

    if errors:
        raise HTTPException(status_code=422, detail=errors)

    return {"updated": updated}
