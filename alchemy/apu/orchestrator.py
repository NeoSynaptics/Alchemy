"""GPU Stack Orchestrator — smart model placement across dual GPUs + RAM.

Core philosophy: VRAM flows like water.
- P0 RESIDENT (voice + GUI clicker) = never evicted.
- P1 USER_ACTIVE (user's current app) = evicted only by P0.
- P2 AGENT (AI background tasks) = yields to P0/P1.
- P3 WARM (in RAM) = promote to VRAM in seconds.
- P4 COLD (on disk) = needs full load time.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

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
        """Initialize monitor and restore frozen baseline models."""
        await self._monitor.start()
        self._started = True

        # Restore frozen baseline (replaces old hardcoded resident loading)
        actions = await self.restore_frozen_baseline()
        for action in actions:
            logger.info("Boot: %s", action)

        logger.info(
            "Stack orchestrator started (mode=%s, models=%d, frozen=%d)",
            self._mode,
            len(self._registry.all_models()),
            sum(len(v) for v in self._frozen_config.values()),
        )

    async def close(self) -> None:
        await self._monitor.close()
        self._started = False

    # --- Status ---

    async def status(self) -> StackStatus:
        snapshot = await self._monitor.snapshot()
        await self._sync_ollama_state(snapshot)
        return StackStatus(
            snapshot=snapshot,
            models=self._registry.all_models(),
            mode=self._mode,
        )

    async def _sync_ollama_state(self, snapshot: HardwareSnapshot) -> None:
        """Reconcile registry with Ollama's actual loaded models."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=2.0)) as client:
                resp = await client.get(f"{self._ollama_host}/api/ps")
                resp.raise_for_status()
                ps_data = resp.json()
        except Exception:
            return  # Can't reach Ollama — skip sync

        # Build set of models Ollama actually has loaded
        ollama_loaded: set[str] = set()
        for m in ps_data.get("models", []):
            name = m.get("name", "")
            # Normalize: strip ":latest" suffix
            if name.endswith(":latest"):
                name = name[: -len(":latest")]
            ollama_loaded.add(name)

        # Update registry to match reality
        for card in self._registry.all_models():
            if card.backend != ModelBackend.OLLAMA:
                continue

            name = card.name
            in_ollama = name in ollama_loaded or f"{name}:latest" in ollama_loaded

            if in_ollama and not card.current_location.is_gpu:
                # Ollama has it loaded but we think it's not on GPU — fix
                # Try to figure out which GPU from VRAM usage
                gpu_idx = card.preferred_gpu if card.preferred_gpu is not None else 0
                location = ModelLocation.GPU_0 if gpu_idx == 0 else ModelLocation.GPU_1
                tier = card.current_tier if card.current_tier.priority <= ModelTier.AGENT.priority else card.default_tier
                self._registry.update_location(name, location, tier)
                logger.debug("Sync: %s found in Ollama, updated to %s", name, location.value)
            elif not in_ollama and card.current_location.is_gpu:
                # We think it's on GPU but Ollama doesn't have it — demote to warm
                self._registry.update_location(name, ModelLocation.CPU_RAM, ModelTier.WARM)
                logger.debug("Sync: %s not in Ollama, demoted to CPU_RAM", name)

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

        # All models can be demoted — core models reload fast from RAM

        if card.current_location.is_gpu:
            await self._backend_unload(card)

        self._registry.update_location(name, ModelLocation.CPU_RAM, ModelTier.WARM)
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

        # GPU models first
        for slot in ("gpu_0", "gpu_1"):
            gpu_index = int(slot[-1])
            for model_name in self._frozen_config.get(slot, []):
                if model_name in skip:
                    continue
                card = self._registry.get(model_name)
                if card is None:
                    continue
                if card.current_location.is_gpu and card.current_location.gpu_index == gpu_index:
                    continue  # Already where it should be
                result = await self.load_model(model_name, gpu=gpu_index)
                if result.success:
                    actions.append(f"Restored {model_name} → GPU {gpu_index}")
                else:
                    actions.append(f"Failed to restore {model_name} → GPU {gpu_index}: {result.error}")

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
