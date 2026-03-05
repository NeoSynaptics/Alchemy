"""Contract guard — FastAPI dependency that enforces model contracts at runtime.

Usage in any API route:

    from alchemy.api.contract_guard import require_contract

    @router.post("/review", dependencies=[Depends(require_contract("gate"))])
    async def review_tool_call(...):
        ...

If the module's required models are missing, the endpoint returns 503 with
a clear error explaining which capabilities need to be pulled.
"""

from __future__ import annotations

from fastapi import HTTPException, Request


def require_contract(module_id: str):
    """Create a FastAPI dependency that checks a module's model contract.

    Raises HTTP 503 if required models are not available in the fleet.
    Passes silently if:
    - No contract reports exist (GPU orchestrator didn't start)
    - The module has no model requirements
    - All required models are satisfied
    """

    async def _check(request: Request):
        reports = getattr(request.app.state, "contract_reports", {})
        report = reports.get(module_id)
        if report and not report.satisfied:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "model_contract_unsatisfied",
                    "module": module_id,
                    "missing_capabilities": report.missing,
                    "message": (
                        f"Module '{module_id}' requires models with capabilities: "
                        f"{report.missing}. Pull the required models first."
                    ),
                },
            )

    return _check
