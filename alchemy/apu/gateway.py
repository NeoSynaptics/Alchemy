"""APU Inference Gateway — single point of contact for all LLM calls.

Every module calls gateway.chat() instead of ollama.chat() directly.
The gateway adds: pre-flight model loading, per-model semaphore,
caller tracking, and inference metrics.

OllamaClient handles retry logic and HTTP. The gateway handles
coordination: who gets GPU time, when models load/evict, and
what happened (metrics).
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncGenerator

from alchemy.apu.metrics import InferenceMetrics, InferenceRecord

if TYPE_CHECKING:
    from alchemy.adapters.ollama import OllamaClient
    from alchemy.apu.orchestrator import StackOrchestrator
    from alchemy.apu.registry import ModelRegistry

logger = logging.getLogger(__name__)


class _CallerProxy:
    """Thin proxy that sets default caller/priority on all gateway calls.

    Created via ``gateway.with_caller("module_name", priority=5)``.
    Modules receive this instead of the raw gateway — their code stays
    unchanged while every inference call gets proper metrics attribution.
    """

    __slots__ = ("_gw", "_caller", "_priority")

    def __init__(self, gateway: APUGateway, caller: str, priority: int) -> None:
        self._gw = gateway
        self._caller = caller
        self._priority = priority

    async def chat(self, model, messages, **kwargs):
        kwargs.setdefault("caller", self._caller)
        kwargs.setdefault("priority", self._priority)
        return await self._gw.chat(model, messages, **kwargs)

    async def chat_think(self, model, messages, **kwargs):
        kwargs.setdefault("caller", self._caller)
        kwargs.setdefault("priority", self._priority)
        return await self._gw.chat_think(model, messages, **kwargs)

    async def chat_stream(self, model, messages, **kwargs):
        kwargs.setdefault("caller", self._caller)
        kwargs.setdefault("priority", self._priority)
        return await self._gw.chat_stream(model, messages, **kwargs)

    async def chat_stream_raw(self, model, messages, **kwargs):
        kwargs.setdefault("caller", self._caller)
        kwargs.setdefault("priority", self._priority)
        async for chunk in self._gw.chat_stream_raw(model, messages, **kwargs):
            yield chunk

    async def embed(self, model, text, **kwargs):
        kwargs.setdefault("caller", self._caller)
        kwargs.setdefault("priority", self._priority)
        return await self._gw.embed(model, text, **kwargs)

    def batch_hold(self, model: str):
        return self._gw.batch_hold(model)

    @property
    def ollama(self):
        """Direct access to underlying OllamaClient for non-gatewayed methods."""
        return self._gw.ollama


class APUGateway:
    """Inference gateway — wraps OllamaClient with model management and metrics.

    Modules call gateway methods instead of OllamaClient directly.
    The gateway ensures the requested model is loaded before inference,
    serializes per-model requests (Ollama processes 1 per model), and
    records every call for diagnostics.
    """

    def __init__(
        self,
        ollama: OllamaClient,
        orchestrator: StackOrchestrator | None = None,
        registry: ModelRegistry | None = None,
    ) -> None:
        self._ollama = ollama
        self._orchestrator = orchestrator
        self._registry = registry
        self._model_locks: dict[str, asyncio.Semaphore] = {}
        self._metrics = InferenceMetrics()

    # ── Core inference methods ────────────────────────────────

    async def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        images: list[bytes] | None = None,
        options: dict | None = None,
        think: bool | None = None,
        caller: str = "unknown",
        priority: int = 5,
    ) -> dict:
        """Chat completion through the gateway.

        Same interface as OllamaClient.chat() plus caller/priority tracking.
        Returns full Ollama response dict.
        """
        await self._ensure_model(model, priority, caller)
        sem = self._get_semaphore(model)

        start = time.monotonic()
        success = True
        error_msg = None
        try:
            async with sem:
                return await self._ollama.chat(
                    model, messages, images=images, options=options, think=think,
                )
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            self._record(caller, model, priority, "chat", start, success, error_msg)

    async def chat_think(
        self,
        model: str,
        messages: list[dict],
        *,
        think: bool = True,
        options: dict | None = None,
        seed: int | None = None,
        caller: str = "unknown",
        priority: int = 5,
    ) -> dict:
        """Chat with Qwen3 think mode through the gateway.

        Returns dict with 'content', 'thinking', 'total_duration'.
        """
        await self._ensure_model(model, priority, caller)
        sem = self._get_semaphore(model)

        start = time.monotonic()
        success = True
        error_msg = None
        try:
            async with sem:
                return await self._ollama.chat_think(
                    model, messages, think=think, options=options, seed=seed,
                )
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            self._record(caller, model, priority, "chat_think", start, success, error_msg)

    async def chat_stream(
        self,
        model: str,
        messages: list[dict],
        *,
        images: list[bytes] | None = None,
        options: dict | None = None,
        stop_at: str | None = None,
        caller: str = "unknown",
        priority: int = 5,
    ) -> str:
        """Streaming chat through the gateway. Returns accumulated text."""
        await self._ensure_model(model, priority, caller)
        sem = self._get_semaphore(model)

        start = time.monotonic()
        success = True
        error_msg = None
        try:
            async with sem:
                return await self._ollama.chat_stream(
                    model, messages, images=images, options=options, stop_at=stop_at,
                )
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            self._record(caller, model, priority, "chat_stream", start, success, error_msg)

    async def chat_stream_raw(
        self,
        model: str,
        messages: list[dict],
        *,
        options: dict | None = None,
        think: bool | None = None,
        caller: str = "unknown",
        priority: int = 5,
    ) -> AsyncGenerator[dict, None]:
        """Streaming chat yielding raw Ollama chunks through the gateway.

        Unlike chat_stream (returns accumulated str), this yields individual
        chunks — needed by Voice for real-time token streaming.
        """
        await self._ensure_model(model, priority, caller)
        sem = self._get_semaphore(model)

        start = time.monotonic()
        success = True
        error_msg = None
        await sem.acquire()
        try:
            async for chunk in self._ollama.chat_stream_raw(
                model, messages, options=options, think=think,
            ):
                yield chunk
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            sem.release()
            self._record(caller, model, priority, "chat_stream_raw", start, success, error_msg)

    async def embed(
        self,
        model: str,
        text: str,
        *,
        caller: str = "unknown",
        priority: int = 5,
    ) -> list[float]:
        """Embedding through the gateway. Default priority=3 (WARM)."""
        await self._ensure_model(model, priority, caller)
        sem = self._get_semaphore(model)

        start = time.monotonic()
        success = True
        error_msg = None
        try:
            async with sem:
                return await self._ollama.embed(model, text)
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            self._record(caller, model, priority, "embed", start, success, error_msg)

    # ── Convenience methods (for BaratzaMemory migration) ─────

    async def generate(
        self,
        model: str,
        prompt: str,
        *,
        caller: str = "unknown",
        priority: int = 5,
        options: dict | None = None,
    ) -> str:
        """Simple text generation. Wraps chat() for callers that just need text out.

        Returns the assistant's message content string.
        """
        messages = [{"role": "user", "content": prompt}]
        result = await self.chat(model, messages, caller=caller, priority=priority, options=options)
        return result.get("message", {}).get("content", "")

    async def generate_vision(
        self,
        model: str,
        prompt: str,
        image_b64: str,
        *,
        caller: str = "unknown",
        priority: int = 5,
        options: dict | None = None,
    ) -> str:
        """Vision generation. Takes a base64-encoded image string.

        Converts to bytes for OllamaClient (which re-encodes), then
        returns the assistant's message content string.
        """
        import base64 as b64mod

        image_bytes = b64mod.b64decode(image_b64)
        messages = [{"role": "user", "content": prompt}]
        result = await self.chat(
            model, messages, images=[image_bytes],
            caller=caller, priority=priority, options=options,
        )
        return result.get("message", {}).get("content", "")

    # ── Model management ──────────────────────────────────────

    async def _ensure_model(self, model: str, priority: int, caller: str) -> None:
        """Ensure the requested model is loaded and ready for inference.

        Priority 0 = module disabled (reject call).
        If orchestrator is available, passes caller_priority to ensure_loaded()
        which uses it for eviction decisions.
        """
        if priority == 0:
            raise RuntimeError(f"Module '{caller}' is disabled (priority=0)")
        if self._orchestrator is not None:
            result = await self._orchestrator.ensure_loaded(model, caller_priority=priority)
            if not result.success:
                logger.warning(
                    "[APU Gateway] Failed to load %s for %s (priority=%d): %s",
                    model, caller, priority, result.error,
                )
        elif self._registry is not None:
            card = self._registry.get(model)
            if card:
                card.touch()

    def _get_semaphore(self, model: str) -> asyncio.Semaphore:
        """Get or create a per-model semaphore (1 concurrent request)."""
        if model not in self._model_locks:
            self._model_locks[model] = asyncio.Semaphore(1)
        return self._model_locks[model]

    # ── Metrics ───────────────────────────────────────────────

    def _record(
        self,
        caller: str,
        model: str,
        priority: int,
        method: str,
        start: float,
        success: bool,
        error: str | None,
    ) -> None:
        elapsed_ms = (time.monotonic() - start) * 1000
        rec = InferenceRecord(
            caller=caller,
            model=model,
            priority=priority,
            method=method,
            elapsed_ms=elapsed_ms,
            success=success,
            error=error,
        )
        self._metrics.record(rec)

        # Feed usage pattern predictor
        if self._orchestrator and success:
            self._orchestrator._pattern_predictor.record_call(caller, model)
            # Async pre-warm predicted next models (fire-and-forget)
            predicted = self._orchestrator._pattern_predictor.predict_next_models(caller)
            for pm in predicted:
                card = self._orchestrator._registry.get(pm) if self._orchestrator._registry else None
                if card and not card.current_location.is_gpu:
                    asyncio.ensure_future(self._orchestrator.ensure_loaded(pm))

        if not success:
            logger.warning(
                "[APU Gateway] %s %s/%s FAILED (%.0fms): %s",
                caller, model, method, elapsed_ms, error,
            )
        elif elapsed_ms > 30_000:
            logger.warning(
                "[APU Gateway] %s %s/%s SLOW (%.0fms)",
                caller, model, method, elapsed_ms,
            )

    def get_metrics(self, last_n: int = 50) -> list[dict]:
        """Recent inference metrics for the API."""
        return self._metrics.recent(last_n)

    @property
    def queue_depth(self) -> dict[str, int]:
        """How many callers are waiting per model (0=busy, 1=free)."""
        return {
            model: sem._value
            for model, sem in self._model_locks.items()
        }

    @asynccontextmanager
    async def batch_hold(self, model: str):
        """Context manager: hold a model on GPU for the duration of a batch.

        Usage::
            async with gateway.batch_hold("qwen3:14b"):
                for item in items:
                    await gateway.chat("qwen3:14b", ...)
        """
        if self._orchestrator:
            self._orchestrator.batch_hold(model)
        try:
            yield
        finally:
            if self._orchestrator:
                self._orchestrator.batch_release(model)

    def with_caller(self, caller: str, priority: int = 5) -> _CallerProxy:
        """Return a proxy that tags all calls with the given caller/priority.

        Priority uses the unified 0-10 scale:
          0 = disabled (calls rejected)
          1-9 = active (higher = more important, evicted last)
          10 = nuclear (everything else yields)

        Usage in server.py::

            app.state.gate_reviewer = GateReviewer(
                ollama_client=gateway.with_caller("gate", priority=3),
            )
        """
        return _CallerProxy(self, caller, priority)

    @property
    def ollama(self) -> OllamaClient:
        """Direct access to OllamaClient for methods not yet gatewayed."""
        return self._ollama
