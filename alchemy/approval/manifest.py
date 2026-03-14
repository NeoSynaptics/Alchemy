"""Approval gate manifest — re-exports from alchemy.core."""

from alchemy.manifest import ModuleManifest

MANIFEST = ModuleManifest(
    id="approval",
    name="Approval Gate",
    description="Detect irreversible actions and pause for human confirmation",
    tier="infra",
    requires=["core"],
)
