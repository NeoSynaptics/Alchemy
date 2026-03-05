"""Module discovery API — lists all registered Alchemy modules with contract status."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Request

from alchemy.registry import all_modules, get

router = APIRouter(tags=["modules"])


@router.get("/modules")
async def list_modules(request: Request):
    """Return all discovered module manifests with contract status.

    Each module includes:
    - Full manifest (id, name, description, models, tier, etc.)
    - contract_satisfied: bool — all required models available
    - contract_missing: list[str] — required capabilities not in fleet
    - contract_optional_missing: list[str] — optional capabilities not available
    """
    reports = getattr(request.app.state, "contract_reports", {})
    result = []
    for m in all_modules():
        entry = asdict(m)
        report = reports.get(m.id)
        if report:
            entry["contract_satisfied"] = report.satisfied
            entry["contract_missing"] = report.missing
            entry["contract_optional_missing"] = report.optional_missing
        else:
            entry["contract_satisfied"] = True  # No contract = no requirements
            entry["contract_missing"] = []
            entry["contract_optional_missing"] = []
        result.append(entry)
    return result


@router.get("/modules/{module_id}")
async def get_module(module_id: str, request: Request):
    """Get a single module's manifest and contract status."""
    manifest = get(module_id)
    if not manifest:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Module '{module_id}' not found")

    entry = asdict(manifest)
    reports = getattr(request.app.state, "contract_reports", {})
    report = reports.get(module_id)
    if report:
        entry["contract_satisfied"] = report.satisfied
        entry["contract_missing"] = report.missing
        entry["contract_optional_missing"] = report.optional_missing
        entry["contract_details"] = [
            {
                "capability": r.requirement.capability,
                "required": r.requirement.required,
                "preferred_model": r.requirement.preferred_model,
                "min_tier": r.requirement.min_tier,
                "available": r.available,
                "matched_model": r.model_name,
                "tier_ok": r.tier_ok,
            }
            for r in report.results
        ]
    else:
        entry["contract_satisfied"] = True
        entry["contract_missing"] = []
        entry["contract_optional_missing"] = []
        entry["contract_details"] = []
    return entry
