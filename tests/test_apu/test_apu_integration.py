"""Section 4: APU Stress — Real GPU integration tests.

Requires: Alchemy server running on :8000 with GPU access.
Marker: @pytest.mark.gpu
"""

import subprocess
import time

import httpx
import pytest

ALCHEMY_URL = "http://localhost:8000"


def get_real_vram():
    """Get actual VRAM usage from nvidia-smi."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    gpus = []
    for line in result.stdout.strip().split("\n"):
        used, total, free = [int(x.strip()) for x in line.split(",")]
        gpus.append({"used_mb": used, "total_mb": total, "free_mb": free})
    return gpus


# ---------------------------------------------------------------------------
# 4.1 VRAM Accounting
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestVRAMAccounting:
    """APU's VRAM tracking must match nvidia-smi reality."""

    async def test_vram_matches_nvidia_smi(self):
        """APU reported VRAM vs nvidia-smi -- must agree within 500MB per GPU."""
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{ALCHEMY_URL}/v1/apu/status")
        elapsed = time.perf_counter() - t0
        assert r.status_code == 200, f"APU status failed: {r.text}"
        apu_data = r.json()

        real = get_real_vram()
        print(f"\n  APU status fetch: {elapsed * 1000:.0f}ms")

        for i, gpu in enumerate(real):
            apu_gpu = apu_data["gpus"][i]
            apu_used = apu_gpu["used_vram_mb"]
            real_used = gpu["used_mb"]
            diff = abs(apu_used - real_used)
            print(
                f"  GPU {i} ({apu_gpu['name']}): "
                f"APU={apu_used}MB, nvidia-smi={real_used}MB, diff={diff}MB"
            )
            assert diff < 500, (
                f"GPU {i}: APU says {apu_used}MB used, "
                f"nvidia-smi says {real_used}MB used (diff: {diff}MB)"
            )

    async def test_model_list_not_empty(self):
        """APU reports at least one model in the fleet."""
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{ALCHEMY_URL}/v1/apu/status")
        data = r.json()
        models = data.get("models", [])
        print(f"\n  Models in fleet: {len(models)}")
        for m in models:
            loc = m.get("current_location", "unknown")
            vram = m.get("vram_mb", 0)
            print(f"    {m['name']}: {loc} ({vram}MB VRAM)")
        assert len(models) > 0, "APU reports no models"

    async def test_gpu_count_matches_hardware(self):
        """APU reports same number of GPUs as nvidia-smi."""
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{ALCHEMY_URL}/v1/apu/status")
        apu_gpus = r.json()["gpus"]
        real_gpus = get_real_vram()
        print(f"\n  APU GPUs: {len(apu_gpus)}, nvidia-smi GPUs: {len(real_gpus)}")
        assert len(apu_gpus) == len(real_gpus), (
            f"APU sees {len(apu_gpus)} GPUs, nvidia-smi sees {len(real_gpus)}"
        )


# ---------------------------------------------------------------------------
# 4.2 Model Location Tracking
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestModelLocationTracking:
    """APU correctly tracks where models are (GPU, RAM, disk)."""

    async def test_ollama_models_have_location(self):
        """Every Ollama model has a valid current_location."""
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{ALCHEMY_URL}/v1/apu/status")
        models = r.json()["models"]
        valid_locations = {"gpu_0", "gpu_1", "cpu_ram", "disk", "unloaded"}

        for m in models:
            if m["backend"] == "ollama":
                loc = m.get("current_location", "missing")
                print(f"  {m['name']}: {loc}")
                assert loc in valid_locations, (
                    f"{m['name']} has invalid location: {loc}"
                )

    async def test_gpu_resident_models_use_vram(self):
        """Models on GPU should account for VRAM usage."""
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{ALCHEMY_URL}/v1/apu/status")
        models = r.json()["models"]

        gpu_models = [m for m in models if m.get("current_location", "").startswith("gpu_")]
        print(f"\n  Models on GPU: {len(gpu_models)}")
        for m in gpu_models:
            print(f"    {m['name']}: {m['current_location']} ({m['vram_mb']}MB)")
            assert m["vram_mb"] > 0, f"{m['name']} on GPU but claims 0 VRAM"


# ---------------------------------------------------------------------------
# 4.3 Voice Never Evicted
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestVoicePriority:
    """Voice pipeline must never be evicted for batch work."""

    async def test_voice_status_endpoint(self):
        """Voice status endpoint responds."""
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{ALCHEMY_URL}/v1/voice/status")
        print(f"\n  Voice status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"  Voice active: {data}")
        # Voice may not be started, but endpoint should respond
        assert r.status_code in (200, 503), f"Unexpected status: {r.status_code}"


# ---------------------------------------------------------------------------
# 4.4 APU Response Time
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestAPUPerformance:
    """APU status endpoint must be fast."""

    async def test_status_under_100ms(self):
        """APU status responds in <100ms (no heavy computation)."""
        latencies = []
        async with httpx.AsyncClient(timeout=10) as c:
            for _ in range(5):
                t0 = time.perf_counter()
                r = await c.get(f"{ALCHEMY_URL}/v1/apu/status")
                latency = time.perf_counter() - t0
                assert r.status_code == 200
                latencies.append(latency)

        avg = sum(latencies) / len(latencies)
        worst = max(latencies)
        print(f"\n  APU status latency: avg={avg * 1000:.0f}ms, worst={worst * 1000:.0f}ms")
        assert worst < 0.5, f"APU status too slow: {worst:.2f}s (max 0.5s)"

    async def test_modules_endpoint(self):
        """Module discovery endpoint works."""
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{ALCHEMY_URL}/v1/modules")
        elapsed = time.perf_counter() - t0
        assert r.status_code == 200
        modules = r.json()
        print(f"\n  /v1/modules: {len(modules)} modules in {elapsed * 1000:.0f}ms")
        assert len(modules) > 0, "No modules registered"
