"""GPU + RAM hardware monitor via pynvml and psutil."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# pynvml is optional — mock mode when unavailable
try:
    import pynvml

    _PYNVML_AVAILABLE = True
except ImportError:
    pynvml = None  # type: ignore[assignment]
    _PYNVML_AVAILABLE = False

try:
    import psutil

    _PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    _PSUTIL_AVAILABLE = False


@dataclass
class GPUProcess:
    pid: int
    name: str
    vram_mb: int


@dataclass
class GPUInfo:
    index: int
    name: str
    total_vram_mb: int
    used_vram_mb: int
    free_vram_mb: int
    temperature_c: int
    utilization_pct: int
    processes: list[GPUProcess] = field(default_factory=list)


@dataclass
class RAMInfo:
    total_mb: int
    used_mb: int
    free_mb: int
    available_mb: int


@dataclass
class HardwareSnapshot:
    gpus: list[GPUInfo]
    ram: RAMInfo


def _query_gpus() -> list[GPUInfo]:
    """Synchronous pynvml query — run via asyncio.to_thread."""
    if not _PYNVML_AVAILABLE:
        return []

    gpus: list[GPUInfo] = []
    try:
        count = pynvml.nvmlDeviceGetCount()
    except pynvml.NVMLError:
        return []

    for i in range(count):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_mb = mem.total // (1024 * 1024)
            used_mb = mem.used // (1024 * 1024)
            free_mb = mem.free // (1024 * 1024)

            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except pynvml.NVMLError:
                temp = 0

            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                util_pct = util.gpu
            except pynvml.NVMLError:
                util_pct = 0

            # Per-process VRAM
            processes: list[GPUProcess] = []
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                for p in procs:
                    proc_mem = (p.usedGpuMemory or 0) // (1024 * 1024)
                    try:
                        import psutil as _ps

                        proc_name = _ps.Process(p.pid).name()
                    except Exception:
                        proc_name = f"pid:{p.pid}"
                    processes.append(GPUProcess(pid=p.pid, name=proc_name, vram_mb=proc_mem))
            except pynvml.NVMLError:
                pass

            gpus.append(
                GPUInfo(
                    index=i,
                    name=name,
                    total_vram_mb=total_mb,
                    used_vram_mb=used_mb,
                    free_vram_mb=free_mb,
                    temperature_c=temp,
                    utilization_pct=util_pct,
                    processes=processes,
                )
            )
        except pynvml.NVMLError as e:
            logger.warning("Failed to query GPU %d: %s", i, e)

    return gpus


def _query_ram() -> RAMInfo:
    """Synchronous psutil query."""
    if not _PSUTIL_AVAILABLE:
        return RAMInfo(total_mb=0, used_mb=0, free_mb=0, available_mb=0)

    mem = psutil.virtual_memory()
    return RAMInfo(
        total_mb=int(mem.total // (1024 * 1024)),
        used_mb=int(mem.used // (1024 * 1024)),
        free_mb=int((mem.total - mem.used) // (1024 * 1024)),
        available_mb=int(mem.available // (1024 * 1024)),
    )


class GPUMonitor:
    """Async hardware monitor for VRAM and RAM."""

    def __init__(self, *, mock_gpus: list[GPUInfo] | None = None) -> None:
        self._initialized = False
        self._mock_gpus = mock_gpus

    async def start(self) -> None:
        if self._mock_gpus is not None:
            self._initialized = True
            logger.info("GPUMonitor started (mock mode, %d GPUs)", len(self._mock_gpus))
            return

        if not _PYNVML_AVAILABLE:
            logger.warning("pynvml not available — GPUMonitor running in degraded mode")
            self._initialized = True
            return

        try:
            await asyncio.to_thread(pynvml.nvmlInit)
            self._initialized = True
            driver = await asyncio.to_thread(pynvml.nvmlSystemGetDriverVersion)
            if isinstance(driver, bytes):
                driver = driver.decode("utf-8")
            count = await asyncio.to_thread(pynvml.nvmlDeviceGetCount)
            logger.info("GPUMonitor started (driver=%s, gpus=%d)", driver, count)
        except Exception as e:
            logger.warning("pynvml init failed: %s — running in degraded mode", e)
            self._initialized = True

    async def close(self) -> None:
        if _PYNVML_AVAILABLE and self._mock_gpus is None:
            try:
                await asyncio.to_thread(pynvml.nvmlShutdown)
            except Exception:
                pass
        self._initialized = False

    async def snapshot(self) -> HardwareSnapshot:
        if self._mock_gpus is not None:
            return HardwareSnapshot(gpus=list(self._mock_gpus), ram=_query_ram())

        gpus = await asyncio.to_thread(_query_gpus)
        ram = await asyncio.to_thread(_query_ram)
        return HardwareSnapshot(gpus=gpus, ram=ram)

    async def get_gpu(self, index: int) -> GPUInfo | None:
        snap = await self.snapshot()
        for gpu in snap.gpus:
            if gpu.index == index:
                return gpu
        return None

    async def gpu_count(self) -> int:
        snap = await self.snapshot()
        return len(snap.gpus)
