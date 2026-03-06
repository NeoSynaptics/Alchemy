"""VRAM swap manager — orchestrates GPU model lifecycle.

Single-GPU mode (RTX 4070 12GB): Whisper, Qwen3, and Fish Speech share
one GPU via serialized loading/unloading.

Dual-GPU mode (RTX 5060 Ti + RTX 4070): All models stay resident on
separate GPUs. Every method becomes a no-op.
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from alchemy.voice.fish_speech import FishSpeechProcess
    from alchemy.voice.stt import WhisperSTT

logger = logging.getLogger(__name__)


class GPUMode(str, Enum):
    SINGLE = "single"
    DUAL = "dual"


class VRAMSlot(str, Enum):
    EMPTY = "empty"
    QWEN3 = "qwen3"
    WHISPER = "whisper"
    FISH_SPEECH = "fish_speech"


class VRAMManager:
    """Orchestrates GPU model loading/unloading for single-GPU setups.

    In dual-GPU mode every public method is a no-op (instant return).
    """

    def __init__(
        self,
        mode: GPUMode = GPUMode.SINGLE,
        ollama_host: str = "http://localhost:11434",
        gpu_model: str = "qwen3:14b",
        keep_alive: str = "30m",
    ) -> None:
        self._mode = mode
        self._ollama_host = ollama_host.rstrip("/")
        self._gpu_model = gpu_model
        self._keep_alive = keep_alive
        self._current_slot = VRAMSlot.EMPTY
        self._lock = asyncio.Lock()
        self._client: httpx.AsyncClient | None = None

    @property
    def mode(self) -> GPUMode:
        return self._mode

    @property
    def current_slot(self) -> VRAMSlot:
        return self._current_slot

    @property
    def is_active(self) -> bool:
        """True when VRAM swapping is enabled (single-GPU mode)."""
        return self._mode == GPUMode.SINGLE

    async def start(self) -> None:
        """Initialize httpx client."""
        self._client = httpx.AsyncClient(
            base_url=self._ollama_host,
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

    async def close(self) -> None:
        """Clean up httpx client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Public swap methods — each is a no-op in dual-GPU mode
    # ------------------------------------------------------------------

    async def ensure_stt(self, stt: WhisperSTT) -> None:
        """Prepare VRAM for STT: unload current model, load Whisper."""
        if not self.is_active:
            return
        async with self._lock:
            await self._clear_slot()
            await stt.load()
            self._current_slot = VRAMSlot.WHISPER
            logger.info("VRAM → Whisper STT")

    async def release_stt(self, stt: WhisperSTT) -> None:
        """Free VRAM after STT: unload Whisper."""
        if not self.is_active:
            return
        async with self._lock:
            await stt.unload()
            self._current_slot = VRAMSlot.EMPTY
            logger.info("VRAM → empty (Whisper unloaded)")

    async def ensure_llm(self) -> None:
        """Prepare VRAM for LLM: warm-load Qwen3 via Ollama."""
        if not self.is_active:
            return
        async with self._lock:
            await self._clear_slot()
            await self._preload_qwen3()
            self._current_slot = VRAMSlot.QWEN3
            logger.info("VRAM → Qwen3 (%s)", self._gpu_model)

    async def release_llm(self) -> None:
        """Free VRAM after LLM: evict Qwen3 via keep_alive=0."""
        if not self.is_active:
            return
        async with self._lock:
            await self._unload_qwen3()
            self._current_slot = VRAMSlot.EMPTY
            logger.info("VRAM → empty (Qwen3 unloaded)")

    async def ensure_tts(self, fish: FishSpeechProcess) -> None:
        """Prepare VRAM for TTS: start Fish Speech subprocess."""
        if not self.is_active:
            return
        async with self._lock:
            await self._clear_slot()
            await fish.start()
            self._current_slot = VRAMSlot.FISH_SPEECH
            logger.info("VRAM → Fish Speech TTS")

    async def release_tts(self, fish: FishSpeechProcess) -> None:
        """Free VRAM after TTS: kill Fish Speech subprocess."""
        if not self.is_active:
            return
        async with self._lock:
            await fish.stop()
            self._current_slot = VRAMSlot.EMPTY
            logger.info("VRAM → empty (Fish Speech stopped)")

    async def restore_idle(self) -> None:
        """Return to idle: reload Qwen3 as the resident model."""
        if not self.is_active:
            return
        async with self._lock:
            await self._clear_slot()
            await self._preload_qwen3()
            self._current_slot = VRAMSlot.QWEN3
            logger.info("VRAM → Qwen3 (idle restored)")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _clear_slot(self) -> None:
        """Dispatch unload based on what currently occupies VRAM."""
        if self._current_slot == VRAMSlot.QWEN3:
            await self._unload_qwen3()
        # Whisper and Fish Speech are unloaded via their release_* methods.
        # If we get here with those slots, it means something went wrong —
        # but we still reset the slot to avoid deadlocks.
        self._current_slot = VRAMSlot.EMPTY

    async def _preload_qwen3(self) -> None:
        """Warm-load Qwen3 via an empty Ollama generate with keep_alive."""
        if not self._client:
            return
        try:
            resp = await self._client.post(
                "/api/generate",
                json={
                    "model": self._gpu_model,
                    "prompt": "",
                    "keep_alive": self._keep_alive,
                },
            )
            resp.raise_for_status()
        except Exception:
            logger.exception("Failed to preload Qwen3")

    async def _unload_qwen3(self) -> None:
        """Evict Qwen3 from VRAM via keep_alive=0."""
        if not self._client:
            return
        try:
            resp = await self._client.post(
                "/api/generate",
                json={
                    "model": self._gpu_model,
                    "prompt": "",
                    "keep_alive": "0",
                },
            )
            resp.raise_for_status()
        except Exception:
            logger.exception("Failed to unload Qwen3")
        await asyncio.sleep(2.0)  # Wait for Ollama to fully evict from VRAM
