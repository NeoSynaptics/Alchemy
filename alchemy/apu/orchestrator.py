"""GPU Stack Orchestrator — smart model placement across dual GPUs + RAM.

Core philosophy: VRAM flows like water.
- P0 RESIDENT (voice + GUI clicker) = never evicted.
- P1 USER_ACTIVE (user's current app) = evicted only by P0.
- P2 AGENT (AI background tasks) = yields to P0/P1.
- P3 WARM (in RAM) = promote to VRAM in seconds.
- P4 COLD (on disk) = needs full load time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from alchemy.apu.event_log import APUEventLog
from alchemy.apu.monitor import GPUMonitor, HardwareSnapshot, GPUInfo, RAMInfo
from alchemy.apu.registry import (
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
        self._state_lock = asyncio.Lock()
        self._event_log = APUEventLog()
        self._pending_operations: set[str] = set()

        # Track which apps have activated models
        self._app_models: dict[str, list[str]] = {}
        # App-level priority: 0 = highest (evicted last), 100 = lowest
        self._app_priority: dict[str, int] = {}
        # Frozen baseline config — models that auto-load on boot / task release
        self._frozen_config: dict[str, list[str]] = {
            "gpu_0": [],
            "gpu_1": [],
            "ram": [],
        }
        self._frozen_config_path = Path("config/frozen_baseline.json")
        self._load_frozen_config()
        # Model profiles from Synthetic Analytics
        self._model_profiles: dict = {}
        self._profiles_path = Path("config/model_profiles.json")
        self._load_model_profiles()

        # Known apps with default priorities
        self._default_app_priority: dict[str, int] = {
            "voice": 10,     # Voice pipeline — always responsive
            "click": 15,     # GUI automation — near-real-time
            "core": 20,      # Agent kernel
            "router": 30,    # Request routing
            "research": 40,  # AlchemyBrowser
            "word": 50,      # AlchemyWord
            "memory": 50,    # AlchemyMemory
            "desktop": 50,   # Desktop agent
            "gate": 60,      # Gate reviewer
        }

    # --- App Priority ---

    def get_app_priority(self, app_name: str) -> int:
        """Get an app's priority. 0=highest, 100=lowest."""
        return self._app_priority.get(
            app_name, self._default_app_priority.get(app_name, 50)
        )

    def set_app_priority(self, app_name: str, priority: int) -> None:
        """Set an app's priority. 0=highest, 100=lowest."""
        self._app_priority[app_name] = max(0, min(100, priority))

    def all_app_priorities(self) -> dict[str, int]:
        """Return all known apps with their effective priority."""
        apps: dict[str, int] = {}
        # Defaults first
        for name, prio in self._default_app_priority.items():
            apps[name] = prio
        # Active apps from app_models
        for name in self._app_models:
            if name not in apps:
                apps[name] = 50
        # User overrides
        apps.update(self._app_priority)
        return dict(sorted(apps.items(), key=lambda x: x[1]))

    def set_app_gpu(self, app_name: str, gpu: int | None) -> None:
        """Set preferred GPU for an app. Affects all its models."""
        models = self._app_models.get(app_name, [])
        for model_name in models:
            card = self._registry.get(model_name)
            if card:
                card.preferred_gpu = gpu

    # --- Lifecycle ---

    async def start(self) -> None:
        """Initialize monitor, reconcile with Ollama, and restore frozen baseline."""
        await self._monitor.start()
        self._started = True

        # Reconcile registry with what Ollama actually has loaded
        reconcile_actions = await self.reconcile_on_startup()
        for action in reconcile_actions:
            logger.info("Startup reconcile: %s", action)

        # Restore frozen baseline (replaces old hardcoded resident loading)
        actions = await self.restore_frozen_baseline()
        for action in actions:
            logger.info("Boot: %s", action)

        # Start periodic VRAM reconciliation (every 5 minutes)
        self._reconcile_task = asyncio.create_task(self._periodic_reconcile())

        logger.info(
            "Stack orchestrator started (mode=%s, models=%d, frozen=%d)",
            self._mode,
            len(self._registry.all_models()),
            sum(len(v) for v in self._frozen_config.values()),
        )

    async def close(self) -> None:
        if hasattr(self, "_reconcile_task") and self._reconcile_task:
            self._reconcile_task.cancel()
            try:
                await self._reconcile_task
            except asyncio.CancelledError:
                pass
        await self._monitor.close()
        self._started = False

    async def _periodic_reconcile(self) -> None:
        """Background task: reconcile VRAM state every 5 minutes."""
        while self._started:
            await asyncio.sleep(300)
            try:
                actions = await self.reconcile_vram()
                if actions:
                    logger.info("Periodic reconcile: %d corrections", len(actions))
            except Exception as e:
                logger.warning("Periodic reconcile failed: %s", e)

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
        async with self._state_lock:
            return await self._load_model_unlocked(name, gpu)

    async def _load_model_unlocked(self, name: str, gpu: int | None = None) -> LoadResult:
        """Inner load_model without lock (for internal callers already holding lock)."""
        t0 = time.monotonic()
        card = self._registry.get(name)
        if card is None:
            self._event_log.record("load", model_name=name, success=False, error="Not in registry")
            return LoadResult(success=False, error=f"Model '{name}' not in registry")

        # Wait if model is being operated on
        if name in self._pending_operations:
            self._event_log.record("error", model_name=name, success=False, error="Model busy")
            return LoadResult(success=False, error=f"Model is currently being loaded/unloaded")
        self._pending_operations.add(name)

        try:
            return await self._load_model_inner(name, card, gpu, t0)
        finally:
            self._pending_operations.discard(name)

    async def _load_model_inner(self, name: str, card, gpu: int | None, t0: float) -> LoadResult:
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

        vram_before = self._registry.total_vram_on_gpu(target_gpu)

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
            # Rollback: re-promote evicted models since we failed to use the space
            for evicted_name in evicted:
                evicted_card = self._registry.get(evicted_name)
                if evicted_card:
                    reload_ok = await self._backend_load(evicted_card, target_gpu)
                    if reload_ok:
                        loc = ModelLocation.GPU_0 if target_gpu == 0 else ModelLocation.GPU_1
                        self._registry.update_location(evicted_name, loc, evicted_card.default_tier)
                        logger.info("Rollback: restored %s to GPU %d", evicted_name, target_gpu)
                    else:
                        logger.warning("Rollback: failed to restore %s", evicted_name)
            self._event_log.record(
                "error", model_name=name, gpu_index=target_gpu,
                vram_before_mb=vram_before, vram_expected_mb=card.vram_mb,
                duration_ms=(time.monotonic() - t0) * 1000,
                success=False, error="Backend load failed",
                details={"evicted_then_rolled_back": evicted},
            )
            return LoadResult(success=False, error=f"Backend load failed for {name}")

        location = ModelLocation.GPU_0 if target_gpu == 0 else ModelLocation.GPU_1
        tier = card.default_tier if card.default_tier.priority <= ModelTier.AGENT.priority else ModelTier.AGENT
        self._registry.update_location(name, location, tier)
        card.touch()

        self._event_log.record(
            "load", model_name=name, gpu_index=target_gpu,
            vram_before_mb=vram_before,
            vram_after_mb=self._registry.total_vram_on_gpu(target_gpu),
            vram_expected_mb=card.vram_mb,
            duration_ms=(time.monotonic() - t0) * 1000,
            details={"evicted": evicted},
        )

        return LoadResult(success=True, location=location, evicted=evicted)

    async def unload_model(self, name: str) -> bool:
        """Unload a model from VRAM to disk (cold)."""
        t0 = time.monotonic()
        async with self._state_lock:
            card = self._registry.get(name)
            if card is None:
                return False

            gpu_idx = card.current_location.gpu_index
            if card.current_location.is_gpu:
                await self._backend_unload(card)

            self._registry.update_location(name, ModelLocation.DISK, ModelTier.COLD)
            self._event_log.record(
                "unload", model_name=name, gpu_index=gpu_idx,
                vram_expected_mb=card.vram_mb,
                duration_ms=(time.monotonic() - t0) * 1000,
            )
            return True

    async def promote(self, name: str, gpu: int | None = None) -> LoadResult:
        """Promote a model from RAM/disk → VRAM."""
        return await self.load_model(name, gpu=gpu)

    async def demote(self, name: str) -> bool:
        """Demote a model from VRAM → RAM (stays warm, not cold)."""
        t0 = time.monotonic()
        async with self._state_lock:
            card = self._registry.get(name)
            if card is None:
                return False

            gpu_idx = card.current_location.gpu_index
            if card.current_location.is_gpu:
                await self._backend_unload(card)

            self._registry.update_location(name, ModelLocation.CPU_RAM, ModelTier.WARM)
            self._event_log.record(
                "demote", model_name=name, gpu_index=gpu_idx,
                vram_expected_mb=card.vram_mb,
                duration_ms=(time.monotonic() - t0) * 1000,
            )
            return True

    async def evict_to_disk(self, name: str) -> bool:
        """Move model from VRAM or RAM → disk (cold storage)."""
        return await self.unload_model(name)

    # --- Smart Placement ---

    async def ensure_loaded(self, name: str) -> LoadResult:
        """Ensure a model is on a GPU before inference.

        1. Already on GPU? → done (touch LRU)
        2. Find target GPU (preferred or any with space)
        3. Enough free VRAM? → load directly
        4. Not enough? → evict lowest-priority (app first, core last)
        5. Still not enough? → try other GPU
        6. No GPU works? → fail
        """
        async with self._state_lock:
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
                sorted_gpus = sorted(snap.gpus, key=lambda g: g.free_vram_mb, reverse=True)
                gpus_to_try = [g.index for g in sorted_gpus]

            for target in gpus_to_try:
                result = await self._load_model_unlocked(name, gpu=target)
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
        """Activate models for an app. All modules get USER_ACTIVE (P1).

        Module tier affects eviction ORDER, not eviction immunity:
        - core models are evicted last among P1 peers
        - app models are evicted first
        - All models CAN be evicted when VRAM is needed
        - Evicted models go to RAM (warm), not disk -- fast reload
        """
        loaded: list[str] = []
        errors: list[str] = []

        for model_name in models:
            card = self._registry.get(model_name)
            if card is None:
                errors.append(f"Model '{model_name}' not in registry")
                continue

            result = await self.ensure_loaded(model_name)
            if result.success:
                # All activated models get USER_ACTIVE (unless already RESIDENT by fleet config)
                if card.current_tier != ModelTier.RESIDENT:
                    card.current_tier = ModelTier.USER_ACTIVE
                card.owner_app = app_name
                card.module_tier = module_tier  # used for eviction ordering
                loaded.append(model_name)
            else:
                errors.append(f"{model_name}: {result.error}")

        self._app_models[app_name] = loaded

        self._event_log.record(
            "app_activate", app_name=app_name,
            success=len(loaded) > 0,
            error="; ".join(errors) if errors else None,
            details={"loaded": loaded, "requested": models},
        )

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
        from alchemy.apu.resolver import ModelResolver

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
        """Deactivate an app. Demote its models back to WARM, then restore frozen baseline.

        Models that belonged to the deactivated app are excluded from the
        restore pass — they were just intentionally released.
        """
        model_names = self._app_models.pop(app_name, [])
        demoted: list[str] = []

        for name in model_names:
            card = self._registry.get(name)
            if card is None:
                continue

            card.owner_app = None
            ok = await self.demote(name)
            if ok:
                demoted.append(name)

        # Repopulate frozen baseline models after task release
        # (skip models that were just deactivated — they were released intentionally)
        restore_actions = await self.restore_frozen_baseline(exclude=set(model_names))
        for action in restore_actions:
            logger.info("Post-deactivate restore: %s", action)

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

    # --- Frozen Baseline ---

    def _load_frozen_config(self) -> None:
        """Load frozen baseline from disk. Falls back to fleet defaults if missing."""
        try:
            if self._frozen_config_path.exists():
                data = json.loads(self._frozen_config_path.read_text())
                self._frozen_config = {
                    "gpu_0": data.get("gpu_0", []),
                    "gpu_1": data.get("gpu_1", []),
                    "ram": data.get("ram", []),
                }
                logger.info("Frozen baseline loaded: %s", self._frozen_config_path)
            else:
                # Build default from fleet config: resident models go to their preferred GPU
                for card in self._registry.all_models():
                    if card.default_tier == ModelTier.RESIDENT:
                        key = f"gpu_{card.preferred_gpu}" if card.preferred_gpu is not None else "gpu_1"
                        self._frozen_config[key].append(card.name)
                    elif card.default_tier == ModelTier.WARM:
                        self._frozen_config["ram"].append(card.name)
        except Exception as e:
            logger.warning("Failed to load frozen config: %s", e)

    def _load_model_profiles(self) -> None:
        """Load Synthetic Analytics profiles from disk."""
        try:
            if self._profiles_path.exists():
                self._model_profiles = json.loads(self._profiles_path.read_text())
                logger.info("Loaded %d model profiles", len(self._model_profiles))
        except Exception as e:
            logger.warning("Failed to load model profiles: %s", e)

    def _get_recommended_ctx(self, model_name: str) -> int | None:
        """Get recommended num_ctx from Synthetic Analytics profile."""
        profile = self._model_profiles.get(model_name)
        if profile:
            return profile.get("recommended_ctx")
        return None

    def get_frozen_config(self) -> dict[str, list[str]]:
        """Return current frozen baseline configuration."""
        return {k: list(v) for k, v in self._frozen_config.items()}

    def save_frozen_config(self, config: dict[str, list[str]]) -> None:
        """Save frozen baseline to disk."""
        self._frozen_config = {
            "gpu_0": config.get("gpu_0", []),
            "gpu_1": config.get("gpu_1", []),
            "ram": config.get("ram", []),
        }
        # Validate all model names exist
        for slot, names in self._frozen_config.items():
            for name in names:
                if self._registry.get(name) is None:
                    logger.warning("Frozen config: unknown model '%s' in %s", name, slot)
        try:
            self._frozen_config_path.parent.mkdir(parents=True, exist_ok=True)
            self._frozen_config_path.write_text(
                json.dumps(self._frozen_config, indent=2)
            )
            logger.info("Frozen baseline saved: %s", self._frozen_config_path)
        except Exception as e:
            logger.error("Failed to save frozen config: %s", e)

    async def restore_frozen_baseline(self, exclude: set[str] | None = None) -> list[str]:
        """Load all frozen baseline models into their assigned slots.

        Called on boot and after a task releases models.
        Args:
            exclude: model names to skip (e.g. just-deactivated models).
        Returns list of actions taken.
        """
        actions: list[str] = []
        skip = exclude or set()

        # GPU models — load all in parallel for speed
        async def _load_one(model_name: str, gpu_index: int) -> str:
            result = await self.load_model(model_name, gpu=gpu_index)
            if result.success:
                return f"Restored {model_name} → GPU {gpu_index}"
            return f"Failed to restore {model_name} → GPU {gpu_index}: {result.error}"

        tasks = []
        for slot in ("gpu_0", "gpu_1"):
            gpu_index = int(slot[-1])
            for model_name in self._frozen_config.get(slot, []):
                if model_name in skip:
                    continue
                card = self._registry.get(model_name)
                if card is None:
                    continue
                if card.backend == ModelBackend.CUSTOM:
                    actions.append(f"Skipped {model_name} → GPU {gpu_index} (managed externally)")
                    continue
                if card.current_location.is_gpu and card.current_location.gpu_index == gpu_index:
                    continue  # Already where it should be
                tasks.append(_load_one(model_name, gpu_index))

        if tasks:
            actions.extend(await asyncio.gather(*tasks))

        # RAM models — just make sure they're at least warm (not cold)
        for model_name in self._frozen_config.get("ram", []):
            if model_name in skip:
                continue
            card = self._registry.get(model_name)
            if card is None:
                continue
            if card.current_location != ModelLocation.DISK:
                continue  # Already in RAM or GPU, fine
            self._registry.update_location(model_name, ModelLocation.CPU_RAM, ModelTier.WARM)
            actions.append(f"Warmed {model_name} → CPU RAM")

        return actions

    # --- Health & Reconciliation ---

    async def health_check(self) -> dict:
        """Run a full health check: VRAM drift, Ollama sync, and model state."""
        issues: list[str] = []
        snap = await self._monitor.snapshot()

        # Check VRAM drift: compare registry totals vs nvidia-smi actual usage
        for gpu in snap.gpus:
            registry_used = self._registry.total_vram_on_gpu(gpu.index)
            actual_used = gpu.total_vram_mb - gpu.free_vram_mb
            drift = abs(actual_used - registry_used)
            if drift > 500:  # >500MB drift is suspicious
                issues.append(
                    f"GPU {gpu.index} VRAM drift: registry={registry_used}MB, "
                    f"actual={actual_used}MB (delta={drift}MB)"
                )

        # Check for models the registry thinks are on GPU but Ollama doesn't have loaded
        try:
            ollama_models = await self._ollama_list_running()
        except Exception as e:
            issues.append(f"Cannot reach Ollama: {e}")
            ollama_models = {}

        for card in self._registry.all_models():
            if card.current_location.is_gpu and card.backend == ModelBackend.OLLAMA:
                if card.name not in ollama_models:
                    issues.append(f"{card.name}: registry says GPU but Ollama doesn't have it loaded")

        return {
            "healthy": len(issues) == 0,
            "issues": issues,
            "gpu_count": len(snap.gpus),
            "models_on_gpu": sum(
                1 for c in self._registry.all_models() if c.current_location.is_gpu
            ),
            "models_in_ram": sum(
                1 for c in self._registry.all_models()
                if c.current_location == ModelLocation.CPU_RAM
            ),
        }

    async def reconcile_vram(self) -> list[str]:
        """Reconcile registry state with actual Ollama state. Fix drift."""
        actions: list[str] = []

        try:
            ollama_models = await self._ollama_list_running()
        except Exception as e:
            logger.warning("VRAM reconciliation skipped — cannot reach Ollama: %s", e)
            return [f"Skipped: {e}"]

        async with self._state_lock:
            for card in self._registry.all_models():
                if card.backend != ModelBackend.OLLAMA:
                    continue

                on_gpu_in_registry = card.current_location.is_gpu
                on_gpu_in_ollama = card.name in ollama_models

                if on_gpu_in_registry and not on_gpu_in_ollama:
                    # Registry thinks it's loaded but Ollama doesn't — mark as warm
                    self._registry.update_location(card.name, ModelLocation.CPU_RAM, ModelTier.WARM)
                    actions.append(f"Corrected {card.name}: was GPU in registry, not in Ollama → WARM")
                elif not on_gpu_in_registry and on_gpu_in_ollama:
                    # Ollama has it loaded but registry doesn't know — infer GPU from preferred
                    inferred_gpu = ollama_models.get(card.name)
                    loc = ModelLocation.GPU_0 if (inferred_gpu or 0) == 0 else ModelLocation.GPU_1
                    self._registry.update_location(card.name, loc, card.default_tier)
                    actions.append(f"Discovered {card.name}: loaded in Ollama → GPU {inferred_gpu or 0}")

        self._event_log.record(
            "reconcile",
            details={"corrections": len(actions), "actions": actions},
        )

        if actions:
            for a in actions:
                logger.info("Reconcile: %s", a)
        else:
            logger.debug("VRAM reconciliation: no drift detected")

        return actions

    async def reconcile_on_startup(self) -> list[str]:
        """Called during start() to sync registry with what Ollama actually has loaded."""
        return await self.reconcile_vram()

    async def _ollama_list_running(self) -> dict[str, int | None]:
        """Query Ollama /api/ps for currently loaded models.

        Returns dict of model_name -> gpu_index (None if unknown).
        """
        import httpx

        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
            resp = await client.get(f"{self._ollama_host}/api/ps")
            resp.raise_for_status()
            data = resp.json()
            result: dict[str, int | None] = {}
            for m in data.get("models", []):
                name = m["name"]
                # Try to infer GPU from size_vram vs registry preferred_gpu
                card = self._registry.get(name)
                gpu = card.preferred_gpu if card else None
                result[name] = gpu
            return result

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
        """Evict models from GPU until needed_mb is free.

        No model is immune. Eviction order from eviction_candidates():
        app models first, then infra, then core. Within each: LRU first.
        Evicted models go to RAM (warm) for fast reload, not disk.
        """
        snap = await self._monitor.snapshot()
        gpu = next((g for g in snap.gpus if g.index == gpu_index), None)
        if gpu is None:
            return None

        registry_used = self._registry.total_vram_on_gpu(gpu_index)
        available = gpu.total_vram_mb - registry_used

        if available >= needed_mb:
            return []  # Already enough room

        evicted: list[str] = []
        candidates = self._registry.eviction_candidates(
            gpu_index, app_priorities=self.all_app_priorities(),
        )

        # Pass 1: evict lower-priority models first
        for candidate in candidates:
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

        # Pass 2: evict same-tier and higher-priority models if still short
        # (candidates already sorted: app before core, LRU first)
        for candidate in candidates:
            if candidate.name in evicted:
                continue

            await self._backend_unload(candidate)
            self._registry.update_location(
                candidate.name, ModelLocation.CPU_RAM, ModelTier.WARM
            )
            evicted.append(candidate.name)
            available += candidate.vram_mb

            if available >= needed_mb:
                return evicted

        return None  # Impossible — not enough total VRAM

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
                # Subprocess models (Whisper, Fish Speech) are managed by external
                # processes. We can't load them — only detect if they're already
                # running on GPU via pynvml process list.
                on_gpu = await self._subprocess_on_gpu(card, gpu_index)
                if on_gpu:
                    logger.info("Subprocess model %s: detected on GPU %d", card.name, gpu_index)
                else:
                    logger.info("Subprocess model %s: not detected on GPU — marking as disk", card.name)
                return on_gpu
            else:
                logger.warning("Unknown backend for %s: %s", card.name, card.backend)
                return False
        except Exception as e:
            logger.error("Backend load failed for %s: %s", card.name, e)
            return False

    async def _backend_unload(self, card: ModelCard) -> bool:
        """Actually unload a model via its backend.

        If unload fails, returns False and the caller MUST NOT mark the model
        as CPU_RAM — it is still on the GPU.
        """
        try:
            if card.backend == ModelBackend.OLLAMA:
                ok = await self._ollama_unload(card.name)
                if not ok:
                    self._event_log.record(
                        "error", model_name=card.name, success=False,
                        error="Ollama unload returned False",
                    )
                return ok
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
            self._event_log.record(
                "error", model_name=card.name, success=False,
                error=f"Backend unload exception: {e}",
            )
            return False

    async def _subprocess_on_gpu(self, card: ModelCard, gpu_index: int) -> bool:
        """Check if a subprocess model is actually consuming VRAM on a GPU.

        Maps known model names to their process names and checks pynvml
        process list for meaningful VRAM usage (>10MB to filter noise).
        """
        # Known subprocess model → process name patterns
        _PROCESS_HINTS: dict[str, list[str]] = {
            "whisper-large-v3": ["whisper", "faster-whisper", "faster_whisper"],
            "fish-speech-s1": ["fish", "fish_speech", "fish-speech"],
        }

        snap = await self._monitor.snapshot()
        gpu = next((g for g in snap.gpus if g.index == gpu_index), None)
        if gpu is None:
            return False

        hints = _PROCESS_HINTS.get(card.name, [card.name.replace(":", "-")])
        for proc in getattr(gpu, "processes", []):
            proc_lower = proc.name.lower()
            if proc.vram_mb > 10 and any(h in proc_lower for h in hints):
                return True
        return False

    async def _ollama_load(self, model_name: str) -> bool:
        """Warm-load an Ollama model into VRAM. Uses Synthetic Analytics profile for num_ctx."""
        import httpx

        card = self._registry.get(model_name)
        is_embedding = card and "embedding" in card.capabilities

        # Get optimal num_ctx from profile
        recommended_ctx = self._get_recommended_ctx(model_name)
        options = {}
        if recommended_ctx:
            options["num_ctx"] = recommended_ctx

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
                if is_embedding:
                    payload: dict = {
                        "model": model_name,
                        "input": "warmup",
                        "keep_alive": "24h",
                    }
                    if options:
                        payload["options"] = options
                    resp = await client.post(f"{self._ollama_host}/api/embed", json=payload)
                else:
                    payload = {
                        "model": model_name,
                        "prompt": "",
                        "keep_alive": "24h",
                    }
                    if options:
                        payload["options"] = options
                    resp = await client.post(f"{self._ollama_host}/api/generate", json=payload)
                resp.raise_for_status()
                ctx_info = f" (num_ctx={recommended_ctx})" if recommended_ctx else ""
                logger.info("Ollama model loaded: %s%s", model_name, ctx_info)
                return True
        except Exception as e:
            logger.error("Ollama load failed for %s: %s", model_name, e)
            return False

    async def _ollama_unload(self, model_name: str) -> bool:
        """Unload an Ollama model by setting keep_alive to 0."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
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
