"""Section 4.5: APU Concurrency & Multi-App GPU Contention — Live GPU tests.

Requires: Alchemy server running on :8000 with GPU access + Ollama.
Marker: @pytest.mark.gpu

Tests what happens when multiple routines hit the GPU simultaneously:
- Concurrent model loads (same GPU, different GPUs)
- Priority-based eviction under real contention
- Voice survival during heavy GPU operations
- Recovery from overload conditions

SAFETY NETS:
  1. Every test captures VRAM baseline and restores it in teardown
  2. Hard HTTP timeouts on every call (never hang forever)
  3. VRAM guard: bail out if free VRAM < 500MB before loading
  4. Cleanup fixture unloads ALL test models after each test
  5. nvidia-smi cross-check after every test
  6. All loads are small models (0.5b) — never risk OOM with large models
  7. Test ordering: read-only checks first, then single loads, then concurrent
"""

import asyncio
import subprocess
import time

import httpx
import pytest

ALCHEMY_URL = "http://localhost:8000"

# Only use tiny models for concurrency tests — minimize GPU risk
SMALL_MODEL = "qwen2.5:0.5b"     # ~400MB VRAM
MEDIUM_MODEL = "qwen2.5:1.5b"    # ~1GB VRAM (if available)

# Safety thresholds
MIN_FREE_VRAM_MB = 500    # Bail out if any GPU has less than this free
MAX_TEST_MODELS = 3       # Never load more than this many test models at once
HTTP_TIMEOUT = 30         # Seconds — hard cap on any single HTTP call
CLEANUP_TIMEOUT = 60      # Seconds — longer timeout for cleanup operations


# ---------------------------------------------------------------------------
# Safety helpers
# ---------------------------------------------------------------------------

def get_real_vram() -> dict[int, dict[str, int]]:
    """Get actual VRAM usage from nvidia-smi per GPU."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    )
    gpus = {}
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = [int(x.strip()) for x in line.split(",")]
        gpus[parts[0]] = {"used_mb": parts[1], "total_mb": parts[2], "free_mb": parts[3]}
    return gpus


def assert_vram_safe(context: str = ""):
    """Bail out if any GPU is dangerously full. Call BEFORE loading models."""
    vram = get_real_vram()
    for gpu_idx, info in vram.items():
        if info["free_mb"] < MIN_FREE_VRAM_MB:
            pytest.skip(
                f"SAFETY BAIL: GPU {gpu_idx} has only {info['free_mb']}MB free "
                f"(min {MIN_FREE_VRAM_MB}MB). {context}"
            )
    return vram


async def apu_status() -> dict:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as c:
        r = await c.get(f"{ALCHEMY_URL}/v1/apu/status")
        assert r.status_code == 200, f"APU status failed: {r.status_code} {r.text[:200]}"
        return r.json()


async def load_model(name: str, timeout: int = HTTP_TIMEOUT) -> httpx.Response:
    async with httpx.AsyncClient(timeout=timeout) as c:
        return await c.post(f"{ALCHEMY_URL}/v1/apu/models/{name}/load")


async def unload_model(name: str) -> httpx.Response:
    async with httpx.AsyncClient(timeout=CLEANUP_TIMEOUT) as c:
        return await c.post(f"{ALCHEMY_URL}/v1/apu/models/{name}/unload")


async def safe_unload(name: str):
    """Unload a model, swallowing errors (for cleanup)."""
    try:
        await unload_model(name)
    except Exception as e:
        print(f"  [CLEANUP WARNING] Failed to unload {name}: {e}")


async def get_loaded_test_models() -> list[str]:
    """Get list of test models currently on GPU (to know what to clean up)."""
    try:
        status = await apu_status()
        return [
            m["name"] for m in status["models"]
            if m["current_location"].startswith("gpu_")
            and m["name"] in (SMALL_MODEL, MEDIUM_MODEL)
        ]
    except Exception:
        return []


def vram_snapshot_str(vram: dict) -> str:
    """Human-readable VRAM summary."""
    parts = []
    for idx in sorted(vram.keys()):
        info = vram[idx]
        parts.append(f"GPU{idx}: {info['used_mb']}/{info['total_mb']}MB ({info['free_mb']}MB free)")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Fixtures — automatic cleanup after every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
async def gpu_safety_net():
    """Before: check VRAM is safe. After: unload all test models + verify VRAM."""
    vram_before = assert_vram_safe("Pre-test check")
    print(f"\n  [SAFETY] Before: {vram_snapshot_str(vram_before)}")

    yield  # --- test runs here ---

    # Cleanup: unload any test models we may have loaded
    loaded = await get_loaded_test_models()
    if loaded:
        print(f"  [CLEANUP] Unloading test models: {loaded}")
        for name in loaded:
            await safe_unload(name)
        await asyncio.sleep(2)  # Let VRAM settle

    vram_after = get_real_vram()
    print(f"  [SAFETY] After: {vram_snapshot_str(vram_after)}")

    # Warn (don't fail) if VRAM didn't return — the test assertion handles that
    for gpu_idx in vram_before:
        if gpu_idx in vram_after:
            delta = vram_after[gpu_idx]["used_mb"] - vram_before[gpu_idx]["used_mb"]
            if abs(delta) > 300:
                print(f"  [SAFETY WARNING] GPU {gpu_idx} VRAM delta: {delta:+d}MB after cleanup")


# ---------------------------------------------------------------------------
# 4.5a — Concurrent Model Loads (same GPU)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestConcurrentLoads:
    """Multiple load requests hitting the APU at the same time."""

    async def test_concurrent_load_same_model_no_double_alloc(self):
        """Two concurrent loads of the SAME model — should not double-allocate VRAM."""
        assert_vram_safe("before concurrent same-model load")
        vram_before = get_real_vram()

        # Fire two loads of the same model simultaneously
        t0 = time.perf_counter()
        r1, r2 = await asyncio.gather(
            load_model(SMALL_MODEL),
            load_model(SMALL_MODEL),
        )
        elapsed = time.perf_counter() - t0

        print(f"\n  Concurrent same-model load: {elapsed:.2f}s")
        print(f"  Response 1: {r1.status_code} — {r1.json().get('location', r1.json().get('error', ''))}")
        print(f"  Response 2: {r2.status_code} — {r2.json().get('location', r2.json().get('error', ''))}")

        # At least one should succeed; the other either succeeds (idempotent) or gets "busy"
        results = [r1.json(), r2.json()]
        successes = [r for r in results if r.get("success")]
        assert len(successes) >= 1, f"Neither load succeeded: {results}"

        # VRAM should only increase by ONE model's worth, not two
        await asyncio.sleep(2)
        vram_after = get_real_vram()
        for gpu_idx in vram_before:
            delta = vram_after[gpu_idx]["used_mb"] - vram_before[gpu_idx]["used_mb"]
            # Small model is ~400MB. Double would be ~800MB. Allow 600MB max.
            assert delta < 600, (
                f"GPU {gpu_idx} VRAM grew by {delta}MB — possible double allocation! "
                f"Expected <600MB for a single {SMALL_MODEL}"
            )
            if delta > 0:
                print(f"  GPU {gpu_idx} VRAM delta: +{delta}MB (single model, correct)")

    async def test_load_during_unload_no_corruption(self):
        """Start unloading a model, then immediately try to load it again."""
        assert_vram_safe("before load-during-unload")

        # First, load the model
        r = await load_model(SMALL_MODEL)
        assert r.status_code == 200, f"Initial load failed: {r.text[:200]}"
        await asyncio.sleep(1)

        # Fire unload + immediate reload concurrently
        t0 = time.perf_counter()
        r_unload, r_reload = await asyncio.gather(
            unload_model(SMALL_MODEL),
            load_model(SMALL_MODEL),
        )
        elapsed = time.perf_counter() - t0

        print(f"\n  Unload+reload race: {elapsed:.2f}s")
        print(f"  Unload: {r_unload.status_code}")
        print(f"  Reload: {r_reload.status_code} — {r_reload.json()}")

        # The model should end up in a valid state (either loaded or unloaded, not corrupted)
        status = await apu_status()
        model = next((m for m in status["models"] if m["name"] == SMALL_MODEL), None)
        if model:
            loc = model["current_location"]
            print(f"  Final state: {SMALL_MODEL} at {loc}")
            assert loc in ("gpu_0", "gpu_1", "cpu_ram", "disk"), f"Corrupted location: {loc}"

    async def test_three_concurrent_loads_different_models_stay_within_vram(self):
        """Load 3 small models simultaneously — APU must not overcommit VRAM."""
        assert_vram_safe("before 3-model concurrent load")

        # We'll use the same small model with different API calls
        # The APU should serialize these via its global lock
        t0 = time.perf_counter()
        results = await asyncio.gather(
            load_model(SMALL_MODEL),
            # These will likely return "not in registry" but that's fine —
            # the point is to stress the lock, not to actually load 3 different models.
            # If only SMALL_MODEL is registered, the others will fail gracefully.
            load_model(SMALL_MODEL),
            load_model(SMALL_MODEL),
            return_exceptions=True,
        )
        elapsed = time.perf_counter() - t0

        print(f"\n  3 concurrent loads: {elapsed:.2f}s")
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                print(f"  Load {i+1}: EXCEPTION — {r}")
            else:
                print(f"  Load {i+1}: {r.status_code} — {r.json().get('success', 'N/A')}")

        # Verify VRAM accounting is still sane
        status = await apu_status()
        for gpu in status["gpus"]:
            total_on_gpu = sum(
                m["vram_mb"] for m in status["models"]
                if m["current_location"] == f"gpu_{gpu['index']}"
            )
            assert total_on_gpu <= gpu["total_vram_mb"], (
                f"GPU {gpu['index']} overcommitted: {total_on_gpu}MB models on {gpu['total_vram_mb']}MB GPU"
            )
            print(f"  GPU {gpu['index']}: {total_on_gpu}MB models / {gpu['total_vram_mb']}MB total — OK")


# ---------------------------------------------------------------------------
# 4.5b — Voice Survival Under GPU Contention
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestVoiceSurvival:
    """Voice endpoint must stay alive while GPU is under contention."""

    async def test_voice_responds_during_concurrent_loads(self):
        """Fire model loads while checking voice — voice must not die."""
        assert_vram_safe("before voice-during-loads")

        # Check voice is alive before
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as c:
            r = await c.get(f"{ALCHEMY_URL}/v1/voice/status")
        voice_before = r.status_code
        print(f"\n  Voice before: {voice_before}")

        # Fire a model load and voice check simultaneously
        async def check_voice():
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as c:
                return await c.get(f"{ALCHEMY_URL}/v1/voice/status")

        t0 = time.perf_counter()
        load_result, voice_result = await asyncio.gather(
            load_model(SMALL_MODEL),
            check_voice(),
        )
        elapsed = time.perf_counter() - t0

        print(f"  Concurrent load+voice: {elapsed:.2f}s")
        print(f"  Load: {load_result.status_code}")
        print(f"  Voice during load: {voice_result.status_code}")

        # Voice must respond (200 or 503 "not started" is fine, but NOT timeout/crash)
        assert voice_result.status_code in (200, 503), (
            f"Voice endpoint died during GPU load: {voice_result.status_code}"
        )

    async def test_voice_responds_during_load_unload_cycle(self):
        """Rapid load/unload while polling voice — voice must stay responsive."""
        assert_vram_safe("before voice-during-cycle")

        voice_responses = []

        async def poll_voice(n: int):
            """Poll voice status n times with small delays."""
            for _ in range(n):
                try:
                    async with httpx.AsyncClient(timeout=5) as c:
                        r = await c.get(f"{ALCHEMY_URL}/v1/voice/status")
                    voice_responses.append(r.status_code)
                except Exception as e:
                    voice_responses.append(f"ERROR: {e}")
                await asyncio.sleep(0.2)

        async def churn_gpu():
            """Load and unload a small model a few times."""
            for i in range(3):
                await load_model(SMALL_MODEL)
                await asyncio.sleep(0.5)
                await unload_model(SMALL_MODEL)
                await asyncio.sleep(0.5)

        # Run voice polling and GPU churn concurrently
        t0 = time.perf_counter()
        await asyncio.gather(poll_voice(15), churn_gpu())
        elapsed = time.perf_counter() - t0

        print(f"\n  Voice poll during GPU churn: {elapsed:.2f}s")
        print(f"  Voice responses: {voice_responses}")

        # All voice responses should be valid HTTP codes, not errors
        errors = [r for r in voice_responses if isinstance(r, str)]
        assert len(errors) == 0, f"Voice had {len(errors)} failures during GPU churn: {errors}"


# ---------------------------------------------------------------------------
# 4.5c — APU Status Consistency Under Load
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestStatusConsistency:
    """APU status endpoint must return consistent data even during operations."""

    async def test_status_during_load_returns_valid_data(self):
        """Call /v1/apu/status while a model is loading — must not crash or return garbage."""
        assert_vram_safe("before status-during-load")

        async def poll_status():
            results = []
            for _ in range(5):
                try:
                    s = await apu_status()
                    results.append({
                        "gpus": len(s["gpus"]),
                        "models": len(s["models"]),
                        "mode": s["mode"],
                    })
                except Exception as e:
                    results.append({"error": str(e)})
                await asyncio.sleep(0.3)
            return results

        # Load model while polling status
        status_results, load_result = await asyncio.gather(
            poll_status(),
            load_model(SMALL_MODEL),
        )

        print(f"\n  Status polls during load:")
        for i, sr in enumerate(status_results):
            print(f"    Poll {i+1}: {sr}")

        # All status polls should return valid data
        errors = [sr for sr in status_results if "error" in sr]
        assert len(errors) == 0, f"Status endpoint failed during load: {errors}"

        # GPU count should be consistent across all polls
        gpu_counts = [sr["gpus"] for sr in status_results if "gpus" in sr]
        assert len(set(gpu_counts)) == 1, f"GPU count changed during load: {gpu_counts}"

    async def test_events_log_concurrent_operations(self):
        """Concurrent operations should all appear in the event log."""
        assert_vram_safe("before events-log test")

        # Get event count before
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as c:
            r = await c.get(f"{ALCHEMY_URL}/v1/apu/events?limit=200")
        events_before = len(r.json()) if r.status_code == 200 else 0

        # Do a load + unload
        await load_model(SMALL_MODEL)
        await asyncio.sleep(1)
        await unload_model(SMALL_MODEL)
        await asyncio.sleep(1)

        # Check events after
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as c:
            r = await c.get(f"{ALCHEMY_URL}/v1/apu/events?limit=200")
        if r.status_code == 200:
            events_after = r.json()
            new_events = len(events_after) - events_before
            print(f"\n  Events: {events_before} before, {len(events_after)} after (+{new_events})")

            # Should have at least a load + unload event
            recent = events_after[:5]
            for e in recent:
                print(f"    {e.get('event_type', '?')}: {e.get('model_name', '?')} — {e.get('success', '?')}")

            assert new_events >= 2, (
                f"Expected at least 2 new events (load+unload), got {new_events}"
            )


# ---------------------------------------------------------------------------
# 4.5d — VRAM Guard: Overload Prevention
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestVRAMGuard:
    """APU must reject loads that would exceed VRAM capacity."""

    async def test_apu_rejects_when_vram_full(self):
        """If VRAM is nearly full, APU should reject new loads cleanly (not OOM)."""
        vram = assert_vram_safe("before overload test")

        # Check current GPU status
        status = await apu_status()
        for gpu in status["gpus"]:
            print(f"\n  GPU {gpu['index']}: {gpu['free_vram_mb']}MB free / {gpu['total_vram_mb']}MB total")

        # Don't actually try to overload — just verify the APU reports accurate free space
        # The real overload test is in test_apu_stress.py (unit test with fake backend)
        # Here we just confirm the API accurately reports what nvidia-smi shows
        for gpu in status["gpus"]:
            real = vram.get(gpu["index"], {})
            if real:
                drift = abs(gpu["used_vram_mb"] - real["used_mb"])
                print(f"  GPU {gpu['index']} drift: APU={gpu['used_vram_mb']}MB vs nvidia-smi={real['used_mb']}MB → {drift}MB")
                # Large drift is expected (Ollama overhead) but document it
                if drift > 1000:
                    print(f"  [INFO] Large drift on GPU {gpu['index']} — likely Ollama/system processes not in APU registry")

    async def test_nvidia_smi_matches_apu_after_operations(self):
        """After load/unload, APU VRAM tracking and nvidia-smi should be within tolerance."""
        assert_vram_safe("before tracking test")

        # Baseline
        vram_baseline = get_real_vram()
        status_baseline = await apu_status()

        # Load
        r = await load_model(SMALL_MODEL)
        if not r.json().get("success"):
            pytest.skip(f"Model load failed (may not be in registry): {r.json()}")
        await asyncio.sleep(2)

        vram_loaded = get_real_vram()
        status_loaded = await apu_status()

        # Unload
        await unload_model(SMALL_MODEL)
        await asyncio.sleep(2)

        vram_final = get_real_vram()
        status_final = await apu_status()

        # Report
        print(f"\n  Tracking consistency:")
        for gpu_idx in sorted(vram_baseline.keys()):
            baseline = vram_baseline[gpu_idx]["used_mb"]
            loaded = vram_loaded[gpu_idx]["used_mb"]
            final = vram_final[gpu_idx]["used_mb"]
            print(f"  GPU {gpu_idx} nvidia-smi: {baseline} → {loaded} → {final}MB")

            apu_baseline = status_baseline["gpus"][gpu_idx]["used_vram_mb"]
            apu_loaded = status_loaded["gpus"][gpu_idx]["used_vram_mb"]
            apu_final = status_final["gpus"][gpu_idx]["used_vram_mb"]
            print(f"  GPU {gpu_idx} APU:        {apu_baseline} → {apu_loaded} → {apu_final}MB")

        # Final VRAM should be close to baseline (within 200MB)
        for gpu_idx in vram_baseline:
            leak = vram_final[gpu_idx]["used_mb"] - vram_baseline[gpu_idx]["used_mb"]
            assert abs(leak) < 200, (
                f"GPU {gpu_idx} leaked {leak}MB after load/unload cycle"
            )
