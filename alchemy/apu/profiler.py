"""Synthetic Analytics — automatic model VRAM profiler.

Loads a model at different num_ctx values, measures actual VRAM usage
via Ollama /api/ps, and saves profiles to config/model_profiles.json.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

DEFAULT_CTX_SIZES = [512, 1024, 2048, 4096, 8192, 16384, 32768]
PROFILES_PATH = Path("config/model_profiles.json")


@dataclass
class CtxProfile:
    num_ctx: int
    vram_mb: int
    total_mb: int  # vram + ram spillover
    load_time_s: float
    success: bool = True


@dataclass
class ModelProfile:
    name: str
    tested_contexts: list[CtxProfile] = field(default_factory=list)
    recommended_ctx: int = 2048
    recommended_vram_mb: int = 0
    profiled_at: str = ""


class ModelProfiler:
    """Profiles Ollama models by loading at different context sizes."""

    def __init__(self, ollama_host: str = "http://localhost:11434") -> None:
        self._host = ollama_host
        self._lock = asyncio.Lock()
        self._running: str | None = None  # model currently being profiled
        self._cancel_requested = False
        self._profiles: dict[str, ModelProfile] = {}
        self._load_profiles()

    def _load_profiles(self) -> None:
        if PROFILES_PATH.exists():
            try:
                data = json.loads(PROFILES_PATH.read_text())
                for name, p in data.items():
                    self._profiles[name] = ModelProfile(
                        name=p["name"],
                        tested_contexts=[CtxProfile(**c) for c in p["tested_contexts"]],
                        recommended_ctx=p["recommended_ctx"],
                        recommended_vram_mb=p["recommended_vram_mb"],
                        profiled_at=p.get("profiled_at", ""),
                    )
                logger.info("Loaded %d model profiles from %s", len(self._profiles), PROFILES_PATH)
            except Exception as e:
                logger.warning("Failed to load model profiles: %s", e)

    def _save_profiles(self) -> None:
        PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {name: asdict(p) for name, p in self._profiles.items()}
        PROFILES_PATH.write_text(json.dumps(data, indent=2))
        logger.info("Saved %d model profiles to %s", len(self._profiles), PROFILES_PATH)

    def get_all_profiles(self) -> dict[str, ModelProfile]:
        return dict(self._profiles)

    def get_profile(self, model_name: str) -> ModelProfile | None:
        return self._profiles.get(model_name)

    @property
    def is_running(self) -> bool:
        return self._running is not None

    @property
    def running_model(self) -> str | None:
        return self._running

    def cancel(self) -> bool:
        """Request cancellation of the current profiling run."""
        if self._running:
            self._cancel_requested = True
            logger.info("Profiling cancellation requested for %s", self._running)
            return True
        return False

    async def _unload_all(self, client: httpx.AsyncClient) -> None:
        """Unload all models from Ollama."""
        try:
            resp = await client.get(f"{self._host}/api/ps")
            resp.raise_for_status()
            models = resp.json().get("models", [])
            for m in models:
                name = m["name"]
                # Try generate first, embed for embedding models
                try:
                    await client.post(
                        f"{self._host}/api/generate",
                        json={"model": name, "keep_alive": 0},
                        timeout=10.0,
                    )
                except Exception:
                    try:
                        await client.post(
                            f"{self._host}/api/embed",
                            json={"model": name, "keep_alive": 0},
                            timeout=10.0,
                        )
                    except Exception:
                        pass
            await asyncio.sleep(1)  # Let Ollama release VRAM
        except Exception as e:
            logger.warning("Failed to unload all models: %s", e)

    async def _load_with_ctx(
        self, client: httpx.AsyncClient, model_name: str, num_ctx: int, is_embedding: bool
    ) -> CtxProfile:
        """Load a model with specific num_ctx, measure VRAM, then unload."""
        start = time.monotonic()
        try:
            if is_embedding:
                resp = await client.post(
                    f"{self._host}/api/embed",
                    json={
                        "model": model_name,
                        "input": "warmup",
                        "keep_alive": "5m",
                        "options": {"num_ctx": num_ctx},
                    },
                )
            else:
                resp = await client.post(
                    f"{self._host}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": "",
                        "keep_alive": "5m",
                        "options": {"num_ctx": num_ctx},
                    },
                )
            resp.raise_for_status()
            load_time = time.monotonic() - start

            # Measure VRAM via /api/ps
            ps = await client.get(f"{self._host}/api/ps")
            ps.raise_for_status()
            vram_mb = 0
            total_mb = 0
            for m in ps.json().get("models", []):
                if m["model"] == model_name or m["name"] == model_name:
                    vram_mb = m.get("size_vram", 0) // (1024 * 1024)
                    total_mb = m.get("size", 0) // (1024 * 1024)
                    break

            # Unload
            if is_embedding:
                await client.post(
                    f"{self._host}/api/embed",
                    json={"model": model_name, "keep_alive": 0},
                    timeout=10.0,
                )
            else:
                await client.post(
                    f"{self._host}/api/generate",
                    json={"model": model_name, "keep_alive": 0},
                    timeout=10.0,
                )
            await asyncio.sleep(0.5)

            return CtxProfile(
                num_ctx=num_ctx,
                vram_mb=vram_mb,
                total_mb=total_mb,
                load_time_s=round(load_time, 2),
                success=True,
            )
        except Exception as e:
            load_time = time.monotonic() - start
            logger.warning("Profile %s ctx=%d failed: %s", model_name, num_ctx, e)
            return CtxProfile(
                num_ctx=num_ctx, vram_mb=0, total_mb=0,
                load_time_s=round(load_time, 2), success=False,
            )

    async def profile_model(
        self,
        model_name: str,
        ctx_sizes: list[int] | None = None,
        is_embedding: bool = False,
        gpu_budget_mb: int = 16384,
    ) -> ModelProfile:
        """Profile a model at various context sizes. Returns ModelProfile."""
        if self._lock.locked():
            raise RuntimeError(f"Already profiling {self._running}")

        async with self._lock:
            self._running = model_name
            self._cancel_requested = False
            sizes = ctx_sizes or DEFAULT_CTX_SIZES
            logger.info("Profiling %s at ctx sizes: %s", model_name, sizes)

            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
                # Unload everything first
                await self._unload_all(client)

                results: list[CtxProfile] = []
                for ctx in sorted(sizes):
                    if self._cancel_requested:
                        logger.info("Profiling cancelled for %s", model_name)
                        break
                    logger.info("  Testing %s @ num_ctx=%d ...", model_name, ctx)
                    profile = await self._load_with_ctx(client, model_name, ctx, is_embedding)
                    results.append(profile)
                    logger.info(
                        "  %s @ ctx=%d: %dMB VRAM, %.1fs load, ok=%s",
                        model_name, ctx, profile.vram_mb, profile.load_time_s, profile.success,
                    )
                    if not profile.success:
                        # If it failed, larger ctx sizes will also fail — stop
                        break

                # Pick recommended: largest ctx that fits gpu_budget
                recommended_ctx = sizes[0]
                recommended_vram = 0
                for r in results:
                    if r.success and r.vram_mb <= gpu_budget_mb:
                        recommended_ctx = r.num_ctx
                        recommended_vram = r.vram_mb

                mp = ModelProfile(
                    name=model_name,
                    tested_contexts=results,
                    recommended_ctx=recommended_ctx,
                    recommended_vram_mb=recommended_vram,
                    profiled_at=datetime.now(timezone.utc).isoformat(),
                )

                self._profiles[model_name] = mp
                self._save_profiles()
                self._running = None
                logger.info(
                    "Profile complete: %s → recommended ctx=%d, vram=%dMB",
                    model_name, recommended_ctx, recommended_vram,
                )
                return mp
