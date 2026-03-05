"""Model contract validation — core checks app model requests against the fleet.

Apps declare what they need via ModelRequirement in their manifest.
This module validates those contracts against the ModelRegistry and reports
what's available, what's missing, and whether the app can start.

Usage (in server.py lifespan or setup wizard):
    from alchemy.contracts import validate_contracts, ContractReport
    report = validate_contracts(registry)
    for r in report:
        if not r.satisfied:
            logger.warning("Module %s missing models: %s", r.module_id, r.missing)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from alchemy.manifest import ModelRequirement, ModuleManifest

logger = logging.getLogger(__name__)


@dataclass
class RequirementResult:
    """Result of checking one ModelRequirement against the fleet."""

    requirement: ModelRequirement
    available: bool = False              # A model with this capability exists
    model_name: str | None = None       # Which model matched (None if unavailable)
    tier_ok: bool = True                # Model meets min_tier requirement


@dataclass
class ContractReport:
    """Full validation report for one module's model contract."""

    module_id: str
    module_name: str
    results: list[RequirementResult] = field(default_factory=list)

    @property
    def satisfied(self) -> bool:
        """All required models are available and meet tier requirements."""
        return all(
            r.available and r.tier_ok
            for r in self.results
            if r.requirement.required
        )

    @property
    def missing(self) -> list[str]:
        """Capabilities that are required but not available."""
        return [
            r.requirement.capability
            for r in self.results
            if r.requirement.required and not (r.available and r.tier_ok)
        ]

    @property
    def optional_missing(self) -> list[str]:
        """Capabilities that are optional and not available."""
        return [
            r.requirement.capability
            for r in self.results
            if not r.requirement.required and not r.available
        ]


# Tier ordering for comparison
_TIER_RANK = {"resident": 0, "user_active": 1, "agent": 2, "warm": 3, "cold": 4}


def _tier_meets_minimum(current_tier: str, min_tier: str) -> bool:
    """Check if current_tier is at least as good as min_tier (lower rank = better)."""
    return _TIER_RANK.get(current_tier, 99) <= _TIER_RANK.get(min_tier, 99)


def validate_module_contract(manifest: ModuleManifest, registry) -> ContractReport:
    """Validate a single module's model contract against the registry.

    Args:
        manifest: The module's manifest with model requirements.
        registry: A ModelRegistry instance (from alchemy.gpu).

    Returns:
        ContractReport with per-requirement results.
    """
    report = ContractReport(module_id=manifest.id, module_name=manifest.name)

    for req in manifest.models:
        result = RequirementResult(requirement=req)

        # Try preferred model first
        if req.preferred_model:
            card = registry.get(req.preferred_model)
            if card and req.capability in card.capabilities:
                result.available = True
                result.model_name = card.name
                result.tier_ok = _tier_meets_minimum(
                    card.current_tier.value, req.min_tier
                )
                report.results.append(result)
                continue

        # Fall back to any model with the required capability
        candidates = registry.find_by_capability(req.capability)
        if candidates:
            # Pick the one with best (lowest) tier
            best = min(candidates, key=lambda c: c.current_tier.priority)
            result.available = True
            result.model_name = best.name
            result.tier_ok = _tier_meets_minimum(
                best.current_tier.value, req.min_tier
            )
        else:
            result.available = False
            result.tier_ok = False

        report.results.append(result)

    return report


def validate_contracts(
    registry,
    manifests: list[ModuleManifest] | None = None,
) -> list[ContractReport]:
    """Validate all module model contracts against the fleet.

    Args:
        registry: A ModelRegistry instance.
        manifests: Module manifests to check. If None, discovers all.

    Returns:
        List of ContractReport, one per module that declares models.
    """
    if manifests is None:
        from alchemy.registry import all_modules
        manifests = all_modules()

    reports = []
    for manifest in manifests:
        if not manifest.models:
            continue
        report = validate_module_contract(manifest, registry)
        reports.append(report)

        if report.satisfied:
            logger.info(
                "Contract OK: %s — all %d model requirements met",
                manifest.id, len(report.results),
            )
        else:
            logger.warning(
                "Contract INCOMPLETE: %s — missing: %s",
                manifest.id, report.missing,
            )

    return reports
