"""GPU Monitor tests — mock pynvml + psutil."""

import pytest

from alchemy.gpu.monitor import GPUInfo, GPUMonitor, GPUProcess, HardwareSnapshot, RAMInfo


def _make_gpu(index: int = 0, name: str = "RTX 4070", total: int = 12288,
              used: int = 4000, free: int = 8288) -> GPUInfo:
    return GPUInfo(
        index=index, name=name, total_vram_mb=total,
        used_vram_mb=used, free_vram_mb=free,
        temperature_c=55, utilization_pct=30,
    )


def _make_dual_gpus() -> list[GPUInfo]:
    return [
        _make_gpu(0, "RTX 4070", 12288, 4000, 8288),
        _make_gpu(1, "RTX 5060 Ti", 16384, 6000, 10384),
    ]


@pytest.mark.asyncio
async def test_mock_mode_snapshot():
    """Mock GPUs return correct data."""
    gpus = _make_dual_gpus()
    monitor = GPUMonitor(mock_gpus=gpus)
    await monitor.start()

    snap = await monitor.snapshot()
    assert isinstance(snap, HardwareSnapshot)
    assert len(snap.gpus) == 2
    assert snap.gpus[0].name == "RTX 4070"
    assert snap.gpus[1].name == "RTX 5060 Ti"
    assert snap.gpus[0].total_vram_mb == 12288
    assert snap.gpus[1].total_vram_mb == 16384

    await monitor.close()


@pytest.mark.asyncio
async def test_mock_mode_get_gpu():
    """get_gpu returns correct GPU by index."""
    gpus = _make_dual_gpus()
    monitor = GPUMonitor(mock_gpus=gpus)
    await monitor.start()

    gpu0 = await monitor.get_gpu(0)
    assert gpu0 is not None
    assert gpu0.index == 0
    assert gpu0.name == "RTX 4070"

    gpu1 = await monitor.get_gpu(1)
    assert gpu1 is not None
    assert gpu1.index == 1

    gpu99 = await monitor.get_gpu(99)
    assert gpu99 is None

    await monitor.close()


@pytest.mark.asyncio
async def test_mock_mode_gpu_count():
    """gpu_count returns correct number."""
    gpus = _make_dual_gpus()
    monitor = GPUMonitor(mock_gpus=gpus)
    await monitor.start()

    count = await monitor.gpu_count()
    assert count == 2

    await monitor.close()


@pytest.mark.asyncio
async def test_empty_mock():
    """No GPUs returns empty list."""
    monitor = GPUMonitor(mock_gpus=[])
    await monitor.start()

    snap = await monitor.snapshot()
    assert len(snap.gpus) == 0

    count = await monitor.gpu_count()
    assert count == 0

    await monitor.close()


@pytest.mark.asyncio
async def test_snapshot_includes_ram():
    """Snapshot always includes RAM info."""
    monitor = GPUMonitor(mock_gpus=[])
    await monitor.start()

    snap = await monitor.snapshot()
    assert isinstance(snap.ram, RAMInfo)
    # psutil should return real RAM if available, or zeros in mock
    assert snap.ram.total_mb >= 0

    await monitor.close()


@pytest.mark.asyncio
async def test_gpu_with_processes():
    """GPU processes are tracked correctly."""
    gpu = GPUInfo(
        index=0, name="RTX 4070", total_vram_mb=12288,
        used_vram_mb=5000, free_vram_mb=7288,
        temperature_c=60, utilization_pct=45,
        processes=[
            GPUProcess(pid=1234, name="ollama", vram_mb=3000),
            GPUProcess(pid=5678, name="python", vram_mb=2000),
        ],
    )
    monitor = GPUMonitor(mock_gpus=[gpu])
    await monitor.start()

    snap = await monitor.snapshot()
    assert len(snap.gpus[0].processes) == 2
    assert snap.gpus[0].processes[0].name == "ollama"
    assert snap.gpus[0].processes[0].vram_mb == 3000

    await monitor.close()


@pytest.mark.asyncio
async def test_start_close_lifecycle():
    """Start and close without errors."""
    monitor = GPUMonitor(mock_gpus=[_make_gpu()])
    await monitor.start()
    assert monitor._initialized

    await monitor.close()
    assert not monitor._initialized


@pytest.mark.asyncio
async def test_degraded_mode_no_pynvml(monkeypatch):
    """When pynvml is not available, still works in degraded mode."""
    import alchemy.gpu.monitor as mod
    monkeypatch.setattr(mod, "_PYNVML_AVAILABLE", False)

    monitor = GPUMonitor()
    await monitor.start()
    assert monitor._initialized

    snap = await monitor.snapshot()
    assert len(snap.gpus) == 0  # No pynvml = no GPU data
    assert isinstance(snap.ram, RAMInfo)

    await monitor.close()
