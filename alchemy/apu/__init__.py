"""APU (Alchemy Processing Unit) — smart model placement across dual GPUs + RAM."""

from alchemy.apu.monitor import GPUInfo, GPUMonitor, GPUProcess, HardwareSnapshot, RAMInfo
from alchemy.apu.orchestrator import LoadResult, StackOrchestrator, StackStatus
from alchemy.apu.resolver import ManifestResolution, ModelResolver, ResolvedModel
from alchemy.apu.guard import AlchemyGuard
from alchemy.apu.registry import (
    ModelBackend,
    ModelCard,
    ModelLocation,
    ModelRegistry,
    ModelTier,
)

__all__ = [
    "AlchemyGuard",
    "GPUInfo",
    "GPUMonitor",
    "GPUProcess",
    "HardwareSnapshot",
    "LoadResult",
    "ManifestResolution",
    "ModelBackend",
    "ModelCard",
    "ModelLocation",
    "ModelRegistry",
    "ModelResolver",
    "ModelTier",
    "RAMInfo",
    "ResolvedModel",
    "StackOrchestrator",
    "StackStatus",
]
