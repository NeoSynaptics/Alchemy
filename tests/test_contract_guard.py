"""Tests for the contract guard — API-level enforcement of model contracts."""

from __future__ import annotations

import pytest
from fastapi import APIRouter, Depends, FastAPI
from fastapi.testclient import TestClient

from alchemy.api.contract_guard import require_contract
from alchemy.contracts import ContractReport, RequirementResult
from alchemy.manifest import ModelRequirement


def _make_app_with_guard(module_id: str) -> FastAPI:
    """Create a minimal FastAPI app with a guarded router."""
    app = FastAPI()
    router = APIRouter(dependencies=[Depends(require_contract(module_id))])

    @router.get("/test")
    async def test_endpoint():
        return {"status": "ok"}

    app.include_router(router)
    return app


class TestContractGuard:
    def test_passes_when_no_reports(self):
        """No GPU orchestrator = no reports = pass through (graceful degradation)."""
        app = _make_app_with_guard("gate")
        # Don't set contract_reports at all
        client = TestClient(app)
        resp = client.get("/test")
        assert resp.status_code == 200

    def test_passes_when_contract_satisfied(self):
        app = _make_app_with_guard("gate")
        report = ContractReport(module_id="gate", module_name="Gate Reviewer")
        report.results = [
            RequirementResult(
                requirement=ModelRequirement(capability="reasoning"),
                available=True,
                model_name="qwen3:14b",
            ),
        ]
        app.state.contract_reports = {"gate": report}

        client = TestClient(app)
        resp = client.get("/test")
        assert resp.status_code == 200

    def test_blocks_when_contract_unsatisfied(self):
        app = _make_app_with_guard("desktop")
        report = ContractReport(module_id="desktop", module_name="Desktop Agent")
        report.results = [
            RequirementResult(
                requirement=ModelRequirement(capability="vision"),
                available=False,
            ),
        ]
        app.state.contract_reports = {"desktop": report}

        client = TestClient(app)
        resp = client.get("/test")
        assert resp.status_code == 503
        body = resp.json()["detail"]
        assert body["error"] == "model_contract_unsatisfied"
        assert body["module"] == "desktop"
        assert "vision" in body["missing_capabilities"]

    def test_passes_when_module_not_in_reports(self):
        """Module has no model requirements — not in reports = pass."""
        app = _make_app_with_guard("shadow")
        app.state.contract_reports = {}  # Empty reports

        client = TestClient(app)
        resp = client.get("/test")
        assert resp.status_code == 200

    def test_passes_when_only_optional_missing(self):
        """Optional models missing doesn't block the endpoint."""
        app = _make_app_with_guard("research")
        report = ContractReport(module_id="research", module_name="AlchemyBrowser")
        report.results = [
            RequirementResult(
                requirement=ModelRequirement(capability="reasoning"),
                available=True,
                model_name="qwen3:14b",
            ),
            RequirementResult(
                requirement=ModelRequirement(capability="embedding", required=False),
                available=False,
            ),
        ]
        app.state.contract_reports = {"research": report}

        client = TestClient(app)
        resp = client.get("/test")
        assert resp.status_code == 200

    def test_error_message_lists_missing_capabilities(self):
        """Error response clearly lists what's missing."""
        app = _make_app_with_guard("agent")
        report = ContractReport(module_id="agent", module_name="GUI Agent")
        report.results = [
            RequirementResult(
                requirement=ModelRequirement(capability="reasoning"),
                available=False,
            ),
            RequirementResult(
                requirement=ModelRequirement(capability="vision", required=False),
                available=False,
            ),
        ]
        app.state.contract_reports = {"agent": report}

        client = TestClient(app)
        resp = client.get("/test")
        assert resp.status_code == 503
        body = resp.json()["detail"]
        # Only required capabilities show as missing
        assert "reasoning" in body["missing_capabilities"]
        assert "vision" not in body["missing_capabilities"]
