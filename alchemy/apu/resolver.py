"""Model resolver — turns capability tags into actual model names.

Three resolution modes:
1. PINNED:    Developer set preferred_model="qwen3:14b" → use it directly
2. AUTO-TAG:  Developer set capability="vision" → table resolves to best model
3. COMBO:     Multiple tags → combo table finds VLM, code+reasoning, etc.

Usage:
    from alchemy.apu.resolver import ModelResolver

    resolver = ModelResolver(registry)
    result = resolver.resolve(requirement)
    # result.model_name = "qwen2.5vl:7b"
    # result.resolution = "combo"  # or "pinned", "single_tag", "fallback"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from alchemy.apu.model_table import ALL_TAGS, COMBO_TABLE, SINGLE_TAG_TABLE, ModelCandidate
from alchemy.manifest import ModelRequirement, ModuleManifest

logger = logging.getLogger(__name__)


@dataclass
class ResolvedModel:
    """Result of resolving a ModelRequirement to an actual model."""

    requirement: ModelRequirement
    model_name: str | None = None        # Resolved model name (None if unresolved)
    resolution: str = "unresolved"       # "pinned" | "combo" | "single_tag" | "fallback" | "unresolved"
    candidates: list[str] = field(default_factory=list)  # All candidates considered
    available: bool = False              # Is the resolved model in the fleet?


@dataclass
class ManifestResolution:
    """Full resolution result for an entire module manifest."""

    module_id: str
    models: list[ResolvedModel] = field(default_factory=list)

    @property
    def all_resolved(self) -> bool:
        """All required models were resolved to available fleet models."""
        return all(
            r.available for r in self.models if r.requirement.required
        )

    @property
    def model_names(self) -> list[str]:
        """List of resolved model names (for passing to app_activate)."""
        return [r.model_name for r in self.models if r.model_name and r.available]

    @property
    def missing(self) -> list[str]:
        """Capabilities that could not be resolved."""
        return [
            r.requirement.capability
            for r in self.models
            if r.requirement.required and not r.available
        ]


class ModelResolver:
    """Resolves capability tags to actual model names using the internal table."""

    def __init__(self, registry=None) -> None:
        """
        Args:
            registry: A ModelRegistry instance. If provided, checks that
                      resolved models actually exist in the fleet.
        """
        self._registry = registry

    def resolve(self, requirement: ModelRequirement) -> ResolvedModel:
        """Resolve a single ModelRequirement to a model name.

        Resolution order:
        1. If preferred_model is set → use it ("pinned")
        2. Parse capability into tags → try combo table
        3. Fall back to single-tag table
        4. Last resort: return first model with matching capability from registry
        """
        result = ResolvedModel(requirement=requirement)

        # 1. PINNED — developer explicitly chose a model
        if requirement.preferred_model:
            result.model_name = requirement.preferred_model
            result.resolution = "pinned"
            result.candidates = [requirement.preferred_model]
            result.available = self._is_available(requirement.preferred_model)
            if not result.available:
                # Pinned model not in fleet — try auto-resolve as fallback
                auto = self._resolve_by_tags(requirement.capability)
                if auto:
                    result.candidates.extend(c.name for c in auto)
                    # Find first available from candidates
                    for candidate in auto:
                        if self._is_available(candidate.name):
                            result.model_name = candidate.name
                            result.resolution = "pinned_fallback"
                            result.available = True
                            break
            return result

        # 2. AUTO — resolve by capability tags
        tags = self._parse_tags(requirement.capability)
        candidates = self._resolve_by_tags_set(tags)

        if not candidates:
            candidates = self._resolve_by_tags(requirement.capability)

        if candidates:
            result.candidates = [c.name for c in candidates]
            for candidate in candidates:
                if self._is_available(candidate.name):
                    result.model_name = candidate.name
                    result.resolution = "combo" if len(tags) > 1 else "single_tag"
                    result.available = True
                    return result

            # None available in fleet — pick the top candidate anyway
            result.model_name = candidates[0].name
            result.resolution = "combo" if len(tags) > 1 else "single_tag"
            result.available = False
            return result

        # 3. FALLBACK — ask registry directly by capability
        if self._registry:
            fleet_models = self._registry.find_by_capability(requirement.capability)
            if fleet_models:
                best = min(fleet_models, key=lambda m: m.current_tier.priority)
                result.model_name = best.name
                result.resolution = "fallback"
                result.available = True
                result.candidates = [m.name for m in fleet_models]
                return result

        return result

    def resolve_manifest(self, manifest: ModuleManifest) -> ManifestResolution:
        """Resolve all model requirements in a manifest.

        Returns a ManifestResolution with resolved model names ready for
        app_activate().
        """
        resolution = ManifestResolution(module_id=manifest.id)
        for req in manifest.models:
            resolved = self.resolve(req)
            resolution.models.append(resolved)
        return resolution

    # --- Internal ---

    def _parse_tags(self, capability: str) -> frozenset[str]:
        """Parse a capability string into a set of tags.

        Supports:
            "vision"            → {"vision"}
            "vision+text"       → {"vision", "text"}
            "vision,text"       → {"vision", "text"}
            "vision text"       → {"vision", "text"}
        """
        # Split on +, comma, or space
        raw = capability.replace("+", " ").replace(",", " ").strip()
        tags = frozenset(t.strip().lower() for t in raw.split() if t.strip())
        return tags

    def _resolve_by_tags_set(self, tags: frozenset[str]) -> list[ModelCandidate]:
        """Try combo table with exact tag set match."""
        if len(tags) < 2:
            return []
        candidates = COMBO_TABLE.get(tags, [])
        if candidates:
            return sorted(candidates, key=lambda c: c.score)

        # Try all subsets of size 2+ (most specific first)
        if len(tags) > 2:
            for combo_tags, combo_candidates in COMBO_TABLE.items():
                if combo_tags.issubset(tags):
                    return sorted(combo_candidates, key=lambda c: c.score)

        return []

    def _resolve_by_tags(self, capability: str) -> list[ModelCandidate]:
        """Resolve using single-tag table. If multiple tags, merge and rank."""
        tags = self._parse_tags(capability)

        if len(tags) == 1:
            tag = next(iter(tags))
            return SINGLE_TAG_TABLE.get(tag, [])

        # Multiple tags but no combo match — score by how many tags each model covers
        model_scores: dict[str, int] = {}
        for tag in tags:
            for candidate in SINGLE_TAG_TABLE.get(tag, []):
                if candidate.name not in model_scores:
                    model_scores[candidate.name] = 0
                model_scores[candidate.name] += 1

        if not model_scores:
            return []

        # Sort by: most tags matched (desc), then alphabetical
        ranked = sorted(
            model_scores.items(),
            key=lambda x: (-x[1], x[0]),
        )
        return [ModelCandidate(name=name, score=i + 1) for i, (name, _) in enumerate(ranked)]

    def _is_available(self, model_name: str) -> bool:
        """Check if a model exists in the fleet registry."""
        if self._registry is None:
            return True  # No registry = assume available (testing mode)
        return self._registry.get(model_name) is not None
