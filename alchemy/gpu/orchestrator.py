"""GPU Stack Orchestrator — smart model placement across dual GPUs + RAM.

Core philosophy: VRAM flows like water.
- P0 RESIDENT (voice + GUI clicker) = never evicted.
- P1 USER_ACTIVE (user's current app) = evicted only by P0.
- P2 AGENT (AI background tasks) = yields to P0/P1.
- P3 WARM (in RAM) = promote to VRAM in seconds.
- P4 COLD (on disk) = needs full load time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from alchemy.gpu.monitor import GPUMonitor, HardwareSnapshot, GPUInfo, RAMInfo
from alchemy.gpu.registry import (
    ModelBackend,
    ModelCard,
    ModelLocation,
    ModelRegistry,
    ModelTier,
)

logger = logging.getLogger(__name__)


@dataclass
class StackStatus:
    """Full system status snapshot."""

    snapshot: HardwareSnapshot
    models: list[ModelCard]
    mode: str = "auto"  # "auto" or "manual"


@dataclass
class LoadResult:
    success: bool
    location: ModelLocation | None = None
    evicted: list[str] = field(default_factory=list)
    error: str | None = None


class StackOrchestrator:
    """Manages model placement across GPUs and RAM with priority-based eviction."""

    def __init__(
        self,
        monitor: GPUMonitor,
        registry: ModelRegistry,
        ollama_host: str = "http://localhost:11434",
    ) -> None:
        self._monitor = monitor
        self._registry = registry
        self._ollama_host = ollama_host.rstrip("/")
        self._mode = "auto"
        self._started = False

        # Track which apps have activated models
        self._app_models: dict[str, list[str]] = {}

    # --- Lifecycle ---

    async def start(self) -> None:
        """Initialize monitor and auto-load P0 resident models."""
        await self._monitor.start()
        self._started = True

        # Auto-load residents in order: GPU 1 first (higher VRAM), then GPU 0
        residents = self._registry.models_by_tier(ModelTier.RESIDENT)
        # Sort: preferred_gpu=1 first, then preferred_gpu=0, then None
        residents.sort(key=lambda m: (-(m.preferred_gpu or -1),))

        for model in residents:
            result = await self.load_model(model.name, gpu=model.preferred_gpu)
            if result.success:
                logger.info(
                    "Resident model loaded: %s → %s", model.name, result.location
                )
            else:
                logger.warning(
                    "Failed to load resident model %s: %s", model.name, result.error
                )

        logger.info(
            "Stack orchestrator started (mode=%s, models=%d)",
            self._mode,
            len(self._registry.all_models()),
        )

    async def close(self) -> None:
        await self._monitor.close()
        self._started = False

    # --- Status ---

    async def status(self) -> StackStatus:
        snapshot = await self._monitor.snapshot()
        return StackStatus(
            snapshot=snapshot,
            models=self._registry.all_models(),
            mode=self._mode,
        )

    async def gpu_status(self) -> list[GPUInfo]:
        snap = await self._monitor.snapshot()
        return snap.gpus

    async def ram_status(self) -> RAMInfo:
        snap = await self._monitor.snapshot()
        return snap.ram

    # --- Model Control ---

    async def load_model(self, name: str, gpu: int | None = None) -> LoadResult:
        """Load a model to VRAM. If gpu is None, use preferred_gpu or auto-place."""
        card = self._registry.get(name)
        if card is None:
            return LoadResult(success=False, error=f"Model '{name}' not in registry")

        # Already on a GPU?
        if card.current_location.is_gpu:
            card.touch()
            return LoadResult(success=True, location=card.current_location)

        # Determine target GPU
        target_gpu = gpu if gpu is not None else card.preferred_gpu
        if target_gpu is None:
            target_gpu = await self._auto_select_gpu(card.vram_mb)
            if target_gpu is None:
                return LoadResult(
                    success=False, error="No GPU with enough free VRAM"
                )

        # Ensure enough VRAM (evict if needed)
        evicted = await self._make_room(target_gpu, card.vram_mb, card.current_tier)
        if evicted is None:
            return LoadResult(
                success=False,
                error=f"Cannot free enough VRAM on GPU {target_gpu} for {card.name} ({card.vram_mb}MB)",
            )

        # Perform the actual load
        ok = await self._backend_load(card, target_gpu)
        if not ok:
            return LoadResult(success=False, error=f"Backend load failed for {name}")

        location = ModelLocation.GPU_0 if target_gpu == 0 else ModelLocation.GPU_1
        tier = card.default_tier if card.default_tier.priority <= ModelTier.AGENT.priority else ModelTier.AGENT
        self._registry.update_location(name, location, tier)
        card.touch()

        return LoadResult(success=True, location=location, evicted=evicted)

    async def unload_model(self, name: str) -> bool:
        """Unload a model from VRAM to disk (cold)."""
        card = self._registry.get(name)
        if card is None:
            return False

        if card.current_tier == ModelTier.RESIDENT:
            logger.warning("Refusing to unload P0 resident model: %s", name)
            return False

        if card.current_location.is_gpu:
            await self._backend_unload(card)

        self._registry.update_location(name, ModelLocation.DISK, ModelTier.COLD)
        return True

    async def promote(self, name: str, gpu: int | None = None) -> LoadResult:
        """Promote a model from RAM/disk → VRAM."""
        return await self.load_model(name, gpu=gpu)

    async def demote(self, name: str) -> bool:
        """Demote a model from VRAM → RAM (stays warm, not cold)."""
        card = self._registry.get(name)
        if card is None:
            return False

        if card.current_tier == ModelTier.RESIDENT:
            logger.warning("Refusing to demote P0 resident model: %s", name)
            return False

        if card.current_location.is_gpu:
            await self._backend_unload(card)

        self._registry.update_location(name, ModelLocation.CPU_RAM, ModelTier.WARM)
        return True

    async def evict_to_disk(self, name: str) -> bool:
        """Move model from VRAM or RAM → disk (cold storage)."""
        return await self.unload_model(name)

    # --- Smart Placement ---

    async def ensure_loaded(self, name: str) -> LoadResult:
        """THE key method. Ensure a model is on a GPU before inference.

        1. Already on GPU? → done (touch LRU)
        2. Find target GPU (preferred or any with space)
        3. Enough free VRAM? → load directly
        4. Not enough? → evict lowest-priority (never P0)
        5. Still not enough? → try other GPU
        6. No GPU works? → fail
        """
        card = self._registry.get(name)
        if card is None:
            return LoadResult(success=False, error=f"Model '{name}' not in registry")

        if card.current_location.is_gpu:
            card.touch()
            return LoadResult(success=True, location=card.current_location)

        # Try preferred GPU first, then the other
        gpus_to_try = []
        if card.preferred_gpu is not None:
            gpus_to_try.append(card.preferred_gpu)
            other = 1 - card.preferred_gpu
            gpus_to_try.append(other)
        else:
            snap = await self._monitor.snapshot()
            # Try the GPU with more free VRAM first
            sorted_gpus = sorted(snap.gpus, key=lambda g: g.free_vram_mb, reverse=True)
            gpus_to_try = [g.index for g in sorted_gpus]

        for target in gpus_to_try:
            result = await self.load_model(name, gpu=target)
            if result.success:
                return result

        return LoadResult(
            success=False,
            error=f"Cannot load {name} ({card.vram_mb}MB) — no GPU has enough space",
        )

    async def rebalance(self) -> list[str]:
        """Re-evaluate all placements. Move misplaced models to preferred GPUs."""
        actions: list[str] = []

        for card in self._registry.all_models():
            if not card.current_location.is_gpu:
                continue
            if card.preferred_gpu is None:
                continue

            current_gpu = card.current_location.gpu_index
            if current_gpu == card.preferred_gpu:
                continue

            # Model is on wrong GPU — try to move it
            ok = await self.demote(card.name)
            if ok:
                result = await self.load_model(card.name, gpu=card.preferred_gpu)
                if result.success:
                    actions.append(
                        f"Moved {card.name}: GPU {current_gpu} → GPU {card.preferred_gpu}"
                    )
                else:
                    # Couldn't load on preferred — put it back
                    await self.load_model(card.name, gpu=current_gpu)
                    actions.append(f"Tried to move {card.name} but preferred GPU full")

        return actions

    # --- App Contract ---

    async def app_activate(
        self, app_name: str, models: list[str], module_tier: str = "app",
    ) -> LoadResult:
        """Activate models for an app. Tier-aware priority:

        - core  -> RESIDENT (P0, never evicted)
        - infra -> USER_ACTIVE (P1)
        - app   -> USER_ACTIVE (P1, evictable by core)
        """
        target_tier = ModelTier.RESIDENT if module_tier == "core" else ModelTier.USER_ACTIVE
        loaded: list[str] = []
        errors: list[str] = []

        for model_name in models:
            card = self._registry.get(model_name)
            if card is None:
                errors.append(f"Model '{model_name}' not in registry")
                continue

            result = await self.ensure_loaded(model_name)
            if result.success:
                # Set tier based on module priority (never downgrade RESIDENT)
                if card.current_tier.priority > target_tier.priority:
                    card.current_tier = target_tier
                card.owner_app = app_name
                loaded.append(model_name)
            else:
                errors.append(f"{model_name}: {result.error}")

        self._app_models[app_name] = loaded

        if errors:
            return LoadResult(
                success=len(loaded) > 0,
                error="; ".join(errors),
            )
        return LoadResult(success=True)

    async def app_activate_manifest(self, app_name: str, manifest) -> LoadResult:
        """Activate models for an app using its manifest -- auto-resolves capabilities.

        This is the preferred way to activate an app. The resolver reads the
        manifest's ModelRequirement tags and picks the best models from the fleet.

        Args:
            app_name: Unique app identifier (e.g. "alchemy-word")
            manifest: A ModuleManifest with models=[ModelRequirement(...)]

        Returns:
            LoadResult with success status and any errors.
        """
        from alchemy.gpu.resolver import ModelResolver

        resolver = ModelResolver(self._registry)
        resolution = resolver.resolve_manifest(manifest)

        if not resolution.model_names:
            missing = resolution.missing
            if missing:
                return LoadResult(
                    success=False,
                    error=f"No models available for required capabilities: {missing}",
                )
            return LoadResult(success=True)  # No model requirements

        return await self.app_activate(
            app_name, resolution.model_names, module_tier=manifest.tier,
        )

    async def app_deactivate(self, app_name: str) -> list[str]:
        """Deactivate an app. Demote its models back to WARM (not cold)."""
        model_names = self._app_models.pop(app_name, [])
        demoted: list[str] = []

        for name in model_names:
            card = self._registry.get(name)
            if card is None:
                continue

            # Don't demote P0 residents
            if card.default_tier == ModelTier.RESIDENT:
                continue

            card.owner_app = None
            ok = await self.demote(name)
            if ok:
                demoted.append(name)

        return demoted

    # --- User Activity ---

    async def user_idle(self) -> list[str]:
        """User went idle. Demote all P1 USER_ACTIVE models to WARM."""
        demoted: list[str] = []
        for card in self._registry.models_by_tier(ModelTier.USER_ACTIVE):
            ok = await self.demote(card.name)
            if ok:
                demoted.append(card.name)
        return demoted

    async def user_active(self, app_name: str) -> list[str]:
        """User came back. Re-promote the app's models."""
        model_names = self._app_models.get(app_name, [])
        promoted: list[str] = []
        for name in model_names:
            result = await self.ensure_loaded(name)
            if result.success:
                card = self._registry.get(name)
                if card and card.current_tier != ModelTier.RESIDENT:
                    card.current_tier = ModelTier.USER_ACTIVE
                promoted.append(name)
        return promoted

    # --- Internal Helpers ---

    async def _auto_select_gpu(self, needed_mb: int) -> int | None:
        """Pick the GPU with the most free VRAM that can fit the model."""
        snap = await self._monitor.snapshot()
        best: int | None = None
        best_free = -1

        for gpu in snap.gpus:
            # Account for what the registry says is loaded (more reliable than pynvml alone)
            registry_used = self._registry.total_vram_on_gpu(gpu.index)
            estimated_free = gpu.total_vram_mb - registry_used

            if estimated_free >= needed_mb and estimated_free > best_free:
                best = gpu.index
                best_free = estimated_free

        return best

    async def _make_room(
        self, gpu_index: int, needed_mb: int, requester_tier: ModelTier
    ) -> list[str] | None:
        """Evict models from GPU until needed_mb is free. Returns evicted names, or None if impossible."""
        snap = await self._monitor.snapshot()
        gpu = next((g for g in snap.gpus if g.index == gpu_index), None)
        if gpu is None:
            return None

        registry_used = self._registry.total_vram_on_gpu(gpu_index)
        available = gpu.total_vram_mb - registry_used

        if available >= needed_mb:
            return []  # Already enough room

        evicted: list[str] = []
        candidates = self._registry.eviction_candidates(gpu_index)

        for candidate in candidates:
            # Only evict models with lower priority (higher tier number) than the requester
            if candidate.current_tier.priority <= requester_tier.priority:
                continue

            await self._backend_unload(candidate)
            self._registry.update_location(
                candidate.name, ModelLocation.CPU_RAM, ModelTier.WARM
            )
            evicted.append(candidate.name)
            available += candidate.vram_mb

            if available >= needed_mb:
                return evicted

        # Still not enough — check if evicting same-tier (except P0) works
        for candidate in candidates:
            if candidate.name in evicted:
                continue
            if candidate.current_tier == ModelTier.RESIDENT:
                continue  # NEVER evict P0
            if candidate.current_tier.priority < requester_tier.priority:
                continue  # Don't evict higher-priority models

            await self._backend_unload(candidate)
            self._registry.update_location(
                candidate.name, ModelLocation.CPU_RAM, ModelTier.WARM
            )
            evicted.append(candidate.name)
            available += candidate.vram_mb

            if available >= needed_mb:
                return evicted

        return None  # Impossible

    async def _backend_load(self, card: ModelCard, gpu_index: int) -> bool:
        """Actually load a model via its backend."""
        try:
            if card.backend == ModelBackend.OLLAMA:
                return await self._ollama_load(card.name)
            elif card.backend == ModelBackend.VLLM:
                # vLLM models are managed via process lifecycle — future integration
                logger.info("vLLM model %s: load requested (GPU %d)", card.name, gpu_index)
                return True
            elif card.backend == ModelBackend.SUBPROCESS:
                # Subprocess models (Whisper, Fish Speech) managed externally
                logger.info("Subprocess model %s: marking as loaded", card.name)
                return True
            else:
                logger.warning("Unknown backend for %s: %s", card.name, card.backend)
                return False
        except Exception as e:
            logger.error("Backend load failed for %s: %s", card.name, e)
            return False

    async def _backend_unload(self, card: ModelCard) -> bool:
        """Actually unload a model via its backend."""
        try:
            if card.backend == ModelBackend.OLLAMA:
                return await self._ollama_unload(card.name)
            elif card.backend == ModelBackend.VLLM:
                logger.info("vLLM model %s: unload requested", card.name)
                return True
            elif card.backend == ModelBackend.SUBPROCESS:
                logger.info("Subprocess model %s: marking as unloaded", card.name)
                return True
            else:
                return False
        except Exception as e:
            logger.error("Backend unload failed for %s: %s", card.name, e)
            return False

    async def _ollama_load(self, model_name: str) -> bool:
        """Warm-load an Ollama model by sending a dummy generate request."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
                resp = await client.post(
                    f"{self._ollama_host}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": "",
                        "keep_alive": "30m",
                    },
                )
                resp.raise_for_status()
                logger.info("Ollama model loaded: %s", model_name)
                return True
        except Exception as e:
            logger.error("Ollama load failed for %s: %s", model_name, e)
            return False

    async def _ollama_unload(self, model_name: str) -> bool:
        """Unload an Ollama model by setting keep_alive to 0."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                resp = await client.post(
                    f"{self._ollama_host}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": "",
                        "keep_alive": 0,
                    },
                )
                resp.raise_for_status()
                logger.info("Ollama model unloaded: %s", model_name)
                return True
        except Exception as e:
            logger.error("Ollama unload failed for %s: %s", model_name, e)
            return False
