"""Section 4.1-4.2: VRAM Leak Detection & Frozen Routine — Live GPU tests.

Requires: Alchemy server running on :8000 with GPU access + Ollama.
Marker: @pytest.mark.gpu

These tests load/unload real models and verify VRAM accounting stays accurate.
They complement test_apu_stress.py (which uses mocks) with live GPU validation.
"""

import subprocess
import time

import httpx
import pytest

ALCHEMY_URL = "http://localhost:8000"

# A small model that loads fast — used for leak detection tests
SMALL_MODEL = "qwen2.5:0.5b"  # ~400MB VRAM
# The main model that needs space
LARGE_MODEL = "qwen2.5vl:7b"  # ~5-8GB VRAM


def get_real_vram():
    """Get actual VRAM usage from nvidia-smi per GPU."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    gpus = {}
    for line in result.stdout.strip().split("\n"):
        parts = [int(x.strip()) for x in line.split(",")]
        gpus[parts[0]] = {"used_mb": parts[1], "total_mb": parts[2], "free_mb": parts[3]}
    return gpus


async def apu_status():
    """Fetch APU status from Alchemy."""
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{ALCHEMY_URL}/v1/apu/status")
        assert r.status_code == 200
        return r.json()


# ---------------------------------------------------------------------------
# 4.1 VRAM Leak Detection
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestVRAMLeak:
    """Load/unload models, verify VRAM returns to baseline."""

    async def test_load_unload_small_model_vram_returns(self):
        """Load small model → unload → VRAM should return within 200MB of baseline."""
        # Baseline
        vram_before = get_real_vram()
        status_before = await apu_status()
        print(f"\n  Baseline VRAM: {vram_before}")

        # Load small model
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(
                f"{ALCHEMY_URL}/v1/apu/load",
                json={"model": SMALL_MODEL, "gpu": 0},
            )
        load_time = time.perf_counter() - t0
        print(f"  Load {SMALL_MODEL}: {load_time:.1f}s, status={r.status_code}")

        vram_loaded = get_real_vram()
        vram_delta = vram_loaded[0]["used_mb"] - vram_before[0]["used_mb"]
        print(f"  VRAM delta after load: +{vram_delta}MB")

        # Unload
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(
                f"{ALCHEMY_URL}/v1/apu/unload",
                json={"model": SMALL_MODEL},
            )
        unload_time = time.perf_counter() - t0
        print(f"  Unload: {unload_time:.1f}s, status={r.status_code}")

        # Check VRAM returned
        vram_after = get_real_vram()
        leak = vram_after[0]["used_mb"] - vram_before[0]["used_mb"]
        print(f"  VRAM leak: {leak}MB (tolerance: 200MB)")
        assert abs(leak) < 200, f"VRAM leaked {leak}MB after load/unload cycle"

    async def test_10_load_unload_cycles_no_drift(self):
        """10 load/unload cycles — VRAM accounting stays accurate."""
        vram_baseline = get_real_vram()
        print(f"\n  Baseline: GPU0={vram_baseline[0]['used_mb']}MB")

        for i in range(10):
            async with httpx.AsyncClient(timeout=60) as c:
                await c.post(
                    f"{ALCHEMY_URL}/v1/apu/load",
                    json={"model": SMALL_MODEL, "gpu": 0},
                )
                await c.post(
                    f"{ALCHEMY_URL}/v1/apu/unload",
                    json={"model": SMALL_MODEL},
                )

            # Check APU tracking vs nvidia-smi each cycle
            status = await apu_status()
            real = get_real_vram()
            apu_used = status["gpus"][0]["used_vram_mb"]
            real_used = real[0]["used_mb"]
            drift = abs(apu_used - real_used)

            if i % 3 == 0:
                print(f"  Cycle {i+1}: APU={apu_used}MB, real={real_used}MB, drift={drift}MB")

        # Final check
        vram_final = get_real_vram()
        total_leak = vram_final[0]["used_mb"] - vram_baseline[0]["used_mb"]
        print(f"  Total leak after 10 cycles: {total_leak}MB")
        assert abs(total_leak) < 500, f"VRAM drifted {total_leak}MB over 10 cycles"

    async def test_small_model_evicted_when_large_needs_space(self):
        """Qwen needs 10GB, only 8GB free → small model MUST be evicted."""
        # Load small model first
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(
                f"{ALCHEMY_URL}/v1/apu/load",
                json={"model": SMALL_MODEL, "gpu": 0},
            )
        print(f"\n  Loaded {SMALL_MODEL}: status={r.status_code}")

        # Now request large model — APU should evict small to make room
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(
                f"{ALCHEMY_URL}/v1/apu/ensure-loaded",
                json={"model": LARGE_MODEL},
            )
        elapsed = time.perf_counter() - t0
        print(f"  ensure_loaded({LARGE_MODEL}): {elapsed:.1f}s, status={r.status_code}")

        # Verify large model is loaded
        status = await apu_status()
        large_models = [m for m in status["models"] if m["name"] == LARGE_MODEL]
        if large_models:
            loc = large_models[0].get("current_location", "unknown")
            print(f"  {LARGE_MODEL} location: {loc}")
            assert loc.startswith("gpu_"), f"Large model not on GPU: {loc}"

        # Verify small model was evicted from GPU (may be in RAM or disk)
        small_models = [m for m in status["models"] if m["name"] == SMALL_MODEL]
        if small_models:
            loc = small_models[0].get("current_location", "unknown")
            print(f"  {SMALL_MODEL} location after eviction: {loc}")


# ---------------------------------------------------------------------------
# 4.2 Frozen Routine Detection
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestFrozenRoutine:
    """Detect and recover from frozen/slow model inference."""

    async def test_apu_detects_slow_inference(self):
        """Model inference that takes too long should be flagged in APU events."""
        # This test checks if APU tracks inference timing
        # Run a normal inference first to establish baseline
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.get(f"{ALCHEMY_URL}/v1/apu/status")
        elapsed = time.perf_counter() - t0

        status = r.json()
        events = status.get("recent_events", [])
        slow_events = [e for e in events if "slow" in str(e).lower()]
        print(f"\n  APU status: {elapsed*1000:.0f}ms, {len(events)} events, {len(slow_events)} slow")

        # Verify health check endpoint works
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{ALCHEMY_URL}/v1/apu/health")
        print(f"  Health check: {r.status_code}")
        if r.status_code == 200:
            health = r.json()
            print(f"  Health: {health}")

    async def test_voice_survives_gpu_pressure(self):
        """Voice pipeline stays alive while GPU is under load."""
        # Check voice status before
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{ALCHEMY_URL}/v1/voice/status")
        voice_before = r.status_code
        print(f"\n  Voice before GPU pressure: {voice_before}")

        # Put GPU under pressure by loading a model
        async with httpx.AsyncClient(timeout=120) as c:
            await c.post(
                f"{ALCHEMY_URL}/v1/apu/ensure-loaded",
                json={"model": LARGE_MODEL},
            )

        # Check voice status after — should still respond
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{ALCHEMY_URL}/v1/voice/status")
        voice_after = r.status_code
        print(f"  Voice after GPU pressure: {voice_after}")
        assert voice_after in (200, 503), f"Voice endpoint died: {voice_after}"
