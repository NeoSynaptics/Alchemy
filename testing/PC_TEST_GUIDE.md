# PC Test Implementation Guide

**For:** A Claude window on the PC (dual GPU, Docker, full stack).
**Read first:** `PHILOSOPHY.md` (the WHY), then this file (the HOW).

---

## Your Environment

```
PC Hardware:
  GPU 0: RTX 5060 Ti (16GB) — Qwen 14B, Qwen-VL 7B, Whisper large-v3
  GPU 1: RTX 4070 (12GB) — BGE-M3 (2GB), SigLIP (1.5GB), CLAP (0.5GB)

Docker Services (from BaratzaMemory/docker/docker-compose.yml):
  PostgreSQL 16  — port 5432 (user: baratza, db: baratza)
  Qdrant         — port 6333 (REST), 6334 (gRPC)

Servers:
  Alchemy        — port 8000 (FastAPI)
  NEOSY          — port 8001 (FastAPI)

Repos:
  Alchemy:  C:\Users\monic\Documents\Alchemy_explore  (branch: main)
  NEOSY:    C:\Users\monic\BaratzaMemory               (branch: master)
```

---

## Dev Order (4 Phases)

### Phase 1 — Laptop-Safe (no GPU needed)
The laptop window handles this. Skip unless it hasn't been done.

1. Fix 5 voice timing mocks in `Alchemy/tests/test_voice/`
2. Build the test harness/runner framework (timing, result logging, debug capture)
3. Build the Serpentine test scaffold (Section 9) — empty steps that fill in later

### Phase 2 — PC with Docker (no GPU yet)
Start Docker, run real DB. No model inference needed.

4. **Section 3: Persistence & Recovery** — most critical, data loss = worst bug
5. **Section 2: Buffer & Queue Stress** — mass ingest, concurrent streams

### Phase 3 — PC with GPU
Models loaded, real inference.

6. **Section 1: Size Ladders** — benchmarking text/image/video limits
7. **Section 4: APU Stress** — VRAM leak detection, frozen model recovery
8. **Section 6: Voice Reliability** — command success rates under load
9. **Section 8: NEO Intelligence** — classification quality, cross-language search

### Phase 4 — Polish
Everything wired together.

10. **Section 5: Playwright Scraping** — Instagram saves, scrape→ingest
11. **Section 7: Visual Debugging** — dashboard screenshots, APU visual debugger
12. Wire everything into the Serpentine test (Section 9)
13. Set up nightly run → push results to `testing/results/`

---

## Quick Fixes (Do First)

```bash
# On PC only — install missing dep for research tests
pip install duckduckgo_search
cd C:\Users\monic\Documents\Alchemy_explore
pytest tests/test_research/ -v  # Should fix 8 failures
```

---

## Section 3: Persistence & Recovery — Implementation Spec

**Location:** `BaratzaMemory/tests/test_persistence.py` (NEW)
**Requires:** Docker running (PostgreSQL + Qdrant)
**Marker:** `@pytest.mark.integration`

### What to Build

These tests hit REAL Docker services, not mocks. They verify data survives restarts.

```python
# Pattern for all persistence tests:
#
# 1. Ingest data via the real API (POST http://localhost:8001/ingest)
# 2. Verify it's in PostgreSQL (SELECT from memories)
# 3. Verify it's in Qdrant (query semantic collection)
# 4. Restart Docker / kill process / force error
# 5. Verify data is still there
# 6. Log timing for each step

import pytest
import httpx
import asyncpg
import subprocess
import time

NEOSY_URL = "http://localhost:8001"
PG_DSN = "postgresql://baratza:baratza@localhost:5432/baratza"

@pytest.fixture
async def db_pool():
    pool = await asyncpg.create_pool(PG_DSN)
    yield pool
    await pool.close()

@pytest.fixture
def qdrant():
    from qdrant_client import QdrantClient
    return QdrantClient(host="localhost", port=6333)
```

### 3.1 Restart Survival Tests

```python
class TestRestartSurvival:
    """Data survives Docker restart."""

    async def test_ingest_survives_docker_restart(self, db_pool, qdrant):
        # 1. Ingest 10 items via API
        ids = []
        for i in range(10):
            async with httpx.AsyncClient() as c:
                r = await c.post(f"{NEOSY_URL}/ingest", json={
                    "text": f"Persistence test item {i}: pole vault training",
                    "title": f"Persist #{i}",
                })
                assert r.status_code == 200
                ids.append(r.json()["memory_id"])

        # 2. Verify all in DB
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("SELECT id FROM memories WHERE id = ANY($1::uuid[])", ids)
            assert len(rows) == 10

        # 3. Restart Docker containers
        subprocess.run(["docker", "compose", "-f",
            "C:/Users/monic/BaratzaMemory/docker/docker-compose.yml",
            "restart"], check=True, timeout=60)
        time.sleep(10)  # Wait for services to be healthy

        # 4. Verify all still there
        pool2 = await asyncpg.create_pool(PG_DSN)
        async with pool2.acquire() as conn:
            rows = await conn.fetch("SELECT id FROM memories WHERE id = ANY($1::uuid[])", ids)
            assert len(rows) == 10, f"Lost {10 - len(rows)} memories after restart!"
        await pool2.close()

        # 5. Verify Qdrant vectors survived
        for mid in ids:
            results = qdrant.scroll("semantic", scroll_filter={"must": [
                {"key": "memory_id", "match": {"value": str(mid)}}
            ]}, limit=1)
            assert len(results[0]) > 0, f"Vector missing for {mid} after restart"
```

### 3.2 Transaction Safety Tests

```python
class TestTransactionSafety:
    """Crash between DB write and Qdrant write = DB row exists with status=RAW."""

    async def test_qdrant_failure_leaves_raw_status(self, db_pool):
        # Strategy: Stop Qdrant, ingest, verify DB has row with status=RAW
        subprocess.run(["docker", "compose", "-f",
            "C:/Users/monic/BaratzaMemory/docker/docker-compose.yml",
            "stop", "qdrant"], check=True)

        try:
            async with httpx.AsyncClient() as c:
                r = await c.post(f"{NEOSY_URL}/ingest", json={
                    "text": "This should be saved as RAW",
                    "title": "Transaction safety test",
                })
                # Ingest should still succeed (save to DB, skip Qdrant)
                mid = r.json().get("memory_id")

            if mid:
                async with db_pool.acquire() as conn:
                    row = await conn.fetchrow(
                        "SELECT processing_status FROM memories WHERE id = $1", mid
                    )
                    assert row["processing_status"] == "raw", \
                        f"Expected RAW, got {row['processing_status']}"
        finally:
            # Always restart Qdrant
            subprocess.run(["docker", "compose", "-f",
                "C:/Users/monic/BaratzaMemory/docker/docker-compose.yml",
                "start", "qdrant"], check=True)
            time.sleep(5)
```

---

## Section 2: Buffer & Queue Stress — Implementation Spec

**Location:** `BaratzaMemory/tests/test_stress.py` (NEW)
**Requires:** Docker running
**Marker:** `@pytest.mark.integration`

### What to Build

```python
import asyncio
import time
import httpx
import pytest

NEOSY_URL = "http://localhost:8001"

class TestMassIngest:
    """Massive dumps don't lose data."""

    async def test_100_sequential_no_loss(self):
        """100 items sequentially — all stored, no silent drops."""
        ids = []
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=30) as c:
            for i in range(100):
                r = await c.post(f"{NEOSY_URL}/ingest", json={
                    "text": f"Stress item {i}: {'x' * 200}",
                    "title": f"Stress #{i}",
                })
                assert r.status_code == 200, f"Item {i} failed: {r.text}"
                ids.append(r.json()["memory_id"])
        elapsed = time.perf_counter() - t0

        assert len(set(ids)) == 100, f"Duplicate IDs detected"
        print(f"\n  100 sequential ingests: {elapsed:.1f}s ({elapsed/100*1000:.0f}ms/item)")

    async def test_1000_batch_no_loss(self):
        """1000 items via batch endpoint — all accounted for."""
        items = [{"text": f"Batch stress {i}", "title": f"Batch #{i}"} for i in range(1000)]
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(f"{NEOSY_URL}/ingest/batch", json={
                "items": items, "entity": "user"
            })
        elapsed = time.perf_counter() - t0

        data = r.json()
        assert data["completed"] + data["failed"] == 1000
        assert data["completed"] == 1000, f"{data['failed']} items failed silently"
        print(f"\n  1000 batch ingest: {elapsed:.1f}s ({elapsed/1000*1000:.0f}ms/item)")


class TestConcurrentStreams:
    """Simulate multiple devices pushing data simultaneously."""

    async def test_5_concurrent_streams(self):
        """5 devices, 20 items each, no collisions."""
        all_ids = []
        lock = asyncio.Lock()

        async def device_stream(device_id: int, count: int):
            ids = []
            async with httpx.AsyncClient(timeout=30) as c:
                for i in range(count):
                    r = await c.post(f"{NEOSY_URL}/ingest", json={
                        "text": f"Device {device_id} item {i}",
                        "title": f"Dev{device_id}-{i}",
                    })
                    assert r.status_code == 200
                    ids.append(r.json()["memory_id"])
            async with lock:
                all_ids.extend(ids)

        await asyncio.gather(*[device_stream(d, 20) for d in range(5)])

        assert len(all_ids) == 100
        assert len(set(all_ids)) == 100, "Duplicate memory_ids across streams!"


class TestPriorityPath:
    """Day tasks stay fast even during mass ingest."""

    async def test_search_during_mass_ingest(self):
        """Search responds <500ms while 100 items are being ingested."""
        async def mass_ingest():
            async with httpx.AsyncClient(timeout=60) as c:
                for i in range(100):
                    await c.post(f"{NEOSY_URL}/ingest", json={
                        "text": f"Background item {i}", "title": f"BG-{i}"
                    })

        async def timed_search():
            await asyncio.sleep(1)  # Let ingest start
            async with httpx.AsyncClient(timeout=10) as c:
                t0 = time.perf_counter()
                r = await c.get(f"{NEOSY_URL}/search", params={"text": "pole vault"})
                latency = time.perf_counter() - t0
            assert r.status_code == 200
            assert latency < 0.5, f"Search took {latency:.2f}s during mass ingest (max 0.5s)"
            print(f"\n  Search latency during mass ingest: {latency*1000:.0f}ms")

        await asyncio.gather(mass_ingest(), timed_search())
```

---

## Section 1: Size Ladders — Implementation Spec

**Location:** `BaratzaMemory/tests/test_size_ladder.py` (NEW)
**Requires:** Docker + server running
**Marker:** `@pytest.mark.benchmark`

### What to Build

```python
import time
import httpx
import pytest
import json

NEOSY_URL = "http://localhost:8001"

SIZES_MB = [1, 10, 50, 100, 500]  # Start conservative, add 1GB manually

class TestIngestSizeLadder:
    """Increasing file sizes until the system breaks."""

    @pytest.mark.parametrize("size_mb", SIZES_MB)
    async def test_text_ingest_size(self, size_mb):
        """Ingest {size_mb}MB text, measure time."""
        text = "x" * (size_mb * 1_000_000)
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=300) as c:
            r = await c.post(f"{NEOSY_URL}/ingest", json={
                "text": text,
                "title": f"Size ladder: {size_mb}MB",
            })
        elapsed = time.perf_counter() - t0

        # Log result regardless of pass/fail
        result = {
            "size_mb": size_mb,
            "elapsed_s": round(elapsed, 2),
            "status": r.status_code,
            "mb_per_sec": round(size_mb / elapsed, 2) if elapsed > 0 else 0,
        }
        print(f"\n  SIZE LADDER: {json.dumps(result)}")

        assert r.status_code == 200, f"{size_mb}MB ingest failed: {r.text}"

    # After running all sizes, plot results manually or with:
    # pytest test_size_ladder.py -v -s | grep "SIZE LADDER" > size_results.jsonl
    # Then: python -c "import json,sys; [print(json.loads(l.split('SIZE LADDER: ')[1])) for l in open('size_results.jsonl')]"
```

### Image Size Ladder (Qwen-VL)

```python
from PIL import Image
import io

RESOLUTIONS = [(1000, 1000), (2000, 2000), (4000, 3000), (6000, 4000), (8000, 6000)]

class TestImageSizeLadder:
    """Find where Qwen-VL chokes on image size."""

    @pytest.mark.parametrize("width,height", RESOLUTIONS,
        ids=[f"{w}x{h}" for w,h in RESOLUTIONS])
    async def test_image_classification_size(self, width, height):
        """Classify {width}x{height} image, measure time."""
        # Generate synthetic image
        img = Image.new("RGB", (width, height), color="red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        buf.seek(0)

        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(f"{NEOSY_URL}/ingest", files={
                "file": (f"test_{width}x{height}.jpg", buf, "image/jpeg")
            })
        elapsed = time.perf_counter() - t0

        mp = (width * height) / 1_000_000
        print(f"\n  IMAGE LADDER: {width}x{height} ({mp:.1f}MP) = {elapsed:.1f}s")
        assert r.status_code == 200
```

---

## Section 4: APU Stress — Implementation Spec

**Location:** `Alchemy/tests/test_apu/test_apu_integration.py` (NEW)
**Requires:** Alchemy server running with GPU
**Marker:** `@pytest.mark.gpu`

### What to Build

```python
import httpx
import subprocess
import time
import pytest
import json

ALCHEMY_URL = "http://localhost:8000"

def get_real_vram():
    """Get actual VRAM usage from nvidia-smi."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    gpus = []
    for line in result.stdout.strip().split("\n"):
        used, total = [int(x.strip()) for x in line.split(",")]
        gpus.append({"used_mb": used, "total_mb": total, "free_mb": total - used})
    return gpus

class TestVRAMAccounting:
    """APU's VRAM tracking must match nvidia-smi reality."""

    async def test_vram_matches_nvidia_smi(self):
        """APU reported VRAM vs nvidia-smi — must agree within 200MB."""
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{ALCHEMY_URL}/v1/apu/status")
        apu_data = r.json()

        real = get_real_vram()
        for i, gpu in enumerate(real):
            apu_free = apu_data["gpus"][i]["free_mb"]  # Adjust key to match actual API
            real_free = gpu["free_mb"]
            diff = abs(apu_free - real_free)
            assert diff < 200, (
                f"GPU {i}: APU says {apu_free}MB free, "
                f"nvidia-smi says {real_free}MB free (diff: {diff}MB)"
            )

class TestFrozenModelDetection:
    """APU detects and kills frozen models."""

    async def test_timeout_kills_frozen_model(self):
        """Model that hangs forever → APU fires timeout → VRAM freed."""
        vram_before = get_real_vram()

        # Trigger a model load that will simulate a hang
        # (This depends on APU having a test/debug endpoint or a known slow model)
        # Alternative: load a real model, verify APU tracks it,
        # then verify APU's health guard would flag it if it hung

        async with httpx.AsyncClient() as c:
            status = await c.get(f"{ALCHEMY_URL}/v1/apu/status")

        # Verify health guard is active
        data = status.json()
        assert data.get("health_guard_active", False), "Health guard not running!"

class TestVoiceNeverEvicted:
    """Voice pipeline must never be evicted for batch work."""

    async def test_voice_survives_heavy_batch(self):
        """Load voice + fill GPU with batch work → voice still responds."""
        async with httpx.AsyncClient(timeout=30) as c:
            # 1. Start voice
            await c.post(f"{ALCHEMY_URL}/v1/voice/start")

            # 2. Check voice is running
            r = await c.get(f"{ALCHEMY_URL}/v1/voice/status")
            assert r.json()["active"] is True

            # 3. Trigger heavy GPU work (batch classification)
            # This depends on your batch endpoint — adjust as needed

            # 4. Check voice STILL running
            r = await c.get(f"{ALCHEMY_URL}/v1/voice/status")
            assert r.json()["active"] is True, "Voice was evicted during batch work!"
```

---

## Section 9: Serpentine Scaffold — Implementation Spec

**Location:** `Alchemy/testing/serpentine.py` (NEW — standalone script)
**Requires:** Everything (Docker, both servers, GPU)

### What to Build

```python
#!/usr/bin/env python3
"""The Serpentine Test — one unattended walk through every major path.

Run: python testing/serpentine.py
Output: testing/results/serpentine_{timestamp}.json
"""

import asyncio
import httpx
import json
import time
import subprocess
import sys
from datetime import datetime
from pathlib import Path

NEOSY = "http://localhost:8001"
ALCHEMY = "http://localhost:8000"
RESULTS_DIR = Path(__file__).parent / "results"

class SerpentineRunner:
    def __init__(self):
        self.steps = []
        self.failed = False

    async def step(self, name: str, fn):
        """Run a step, log timing and result."""
        print(f"  [{len(self.steps)+1:2d}] {name}...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            result = await fn()
            elapsed = time.perf_counter() - t0
            self.steps.append({
                "step": len(self.steps) + 1,
                "name": name,
                "status": "PASS",
                "elapsed_s": round(elapsed, 3),
                "detail": result,
            })
            print(f"PASS ({elapsed:.1f}s)")
            return result
        except Exception as e:
            elapsed = time.perf_counter() - t0
            self.steps.append({
                "step": len(self.steps) + 1,
                "name": name,
                "status": "FAIL",
                "elapsed_s": round(elapsed, 3),
                "error": str(e),
            })
            print(f"FAIL ({elapsed:.1f}s) — {e}")
            self.failed = True
            return None

    def save_results(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        RESULTS_DIR.mkdir(exist_ok=True)
        path = RESULTS_DIR / f"serpentine_{ts}.json"
        report = {
            "timestamp": ts,
            "total_steps": len(self.steps),
            "passed": sum(1 for s in self.steps if s["status"] == "PASS"),
            "failed": sum(1 for s in self.steps if s["status"] == "FAIL"),
            "total_time_s": round(sum(s["elapsed_s"] for s in self.steps), 1),
            "steps": self.steps,
        }
        path.write_text(json.dumps(report, indent=2))
        print(f"\n  Results saved to {path}")
        return report


async def main():
    runner = SerpentineRunner()
    memory_ids = []

    # Step 1: Health check
    async def check_health():
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{NEOSY}/health")
            assert r.status_code == 200
            return "NEOSY healthy"

    # Step 2: Ingest 5 text files
    async def ingest_texts():
        ids = []
        async with httpx.AsyncClient(timeout=30) as c:
            for i, text in enumerate([
                "Pole vault approach: 16 steps, 8.2 m/s final speed",
                "Paella valenciana: sofrito, saffron, socarrat",
                "Fibonacci iterative: O(n) time, O(1) space",
                "Neurociencia: el cerebro consume 20% de la energia",
                "3Blue1Brown: vectors are elements of a vector space",
            ]):
                r = await c.post(f"{NEOSY}/ingest", json={"text": text, "title": f"Serpentine #{i+1}"})
                assert r.status_code == 200, f"Ingest {i} failed: {r.text}"
                ids.append(r.json()["memory_id"])
        memory_ids.extend(ids)
        return f"{len(ids)} texts ingested"

    # Step 3: Search for known content
    async def search_pole_vault():
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{NEOSY}/search", params={"text": "pole vault approach speed"})
            assert r.status_code == 200
            results = r.json()
            assert len(results) > 0, "Search returned nothing!"
            return f"Found {len(results)} results"

    # Step 4: Pin a result
    async def pin_result():
        if not memory_ids:
            raise RuntimeError("No memories to pin")
        mid = memory_ids[0]
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(f"{NEOSY}/memories/{mid}/pin", json={"entity": "user", "reason": "Serpentine test"})
            assert r.status_code == 200
            return f"Pinned {mid}"

    # Step 5: Batch ingest 100 items
    async def batch_ingest():
        items = [{"text": f"Batch serpentine item {i}", "title": f"Batch #{i}"} for i in range(100)]
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(f"{NEOSY}/ingest/batch", json={"items": items, "entity": "user"})
            assert r.status_code == 200
            data = r.json()
            assert data["completed"] == 100, f"Only {data['completed']}/100 completed"
            return f"100 items ingested, {data['completed']} completed"

    # Step 6: Search during/after batch (latency check)
    async def search_latency():
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{NEOSY}/search", params={"text": "fibonacci algorithm"})
        latency = time.perf_counter() - t0
        assert latency < 0.5, f"Search too slow: {latency:.2f}s"
        return f"Search latency: {latency*1000:.0f}ms"

    # Step 7: Restart and verify
    async def restart_and_verify():
        subprocess.run(["docker", "compose", "-f",
            "C:/Users/monic/BaratzaMemory/docker/docker-compose.yml",
            "restart"], check=True, timeout=60)
        await asyncio.sleep(15)  # Wait for services

        # Verify data survived
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{NEOSY}/search", params={"text": "pole vault"})
            assert r.status_code == 200
            assert len(r.json()) > 0, "Data lost after restart!"
            return "Data survived restart"

    # Run all steps
    print("\n=== SERPENTINE TEST ===\n")
    await runner.step("Health check", check_health)
    await runner.step("Ingest 5 synthetic texts", ingest_texts)
    await runner.step("Search 'pole vault'", search_pole_vault)
    await runner.step("Pin first result", pin_result)
    await runner.step("Batch ingest 100 items", batch_ingest)
    await runner.step("Search latency check", search_latency)
    await runner.step("Restart Docker + verify persistence", restart_and_verify)

    # TODO: Add when ready:
    # await runner.step("Trigger NEO classification", classify_items)
    # await runner.step("Voice command test", voice_test)
    # await runner.step("Screenshot dashboard", screenshot_dashboard)

    report = runner.save_results()
    print(f"\n  {report['passed']}/{report['total_steps']} passed in {report['total_time_s']}s")

    if runner.failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Result Reporting

All test output goes to `Alchemy/testing/results/`. The format:

```
testing/results/
  alchemy_latest.txt        ← pytest output from Alchemy tests
  neosy_latest.txt          ← pytest output from NEOSY tests
  serpentine_20260311_1430.json  ← Serpentine run with timing per step
  size_ladder_20260311.jsonl    ← Size ladder benchmark data
```

Results are gitignored (regenerated each run). But **TESTING_TODO.md gets updated** with:
- Which checkboxes are now done
- New failure descriptions
- Updated baselines

After each test session:
1. Run tests
2. Read results
3. Update TESTING_TODO.md
4. Commit and push TESTING_TODO.md (not results)

---

## pytest Markers

Add to both repos' `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "integration: requires Docker (PostgreSQL + Qdrant)",
    "benchmark: size ladder and performance tests",
    "gpu: requires GPU for model inference",
    "serpentine: full end-to-end test",
]
```

Run by phase:
```bash
# Phase 2 — Docker only
pytest tests/ -m integration -v

# Phase 3 — GPU required
pytest tests/ -m gpu -v

# Benchmarks only
pytest tests/ -m benchmark -v -s  # -s for print output

# Everything
pytest tests/ -v
```

---

## Critical Rules for PC Window

1. **Never mock what you can test for real.** You have Docker. Use it.
2. **Every test prints timing.** If it doesn't have `time.perf_counter()`, add it.
3. **Silent failure = the worst bug.** If a test passes but data is missing, that's worse than a crash.
4. **Fix the code, not the test.** If a test fails, the system has a bug. Investigate.
5. **Update TESTING_TODO.md** after every session. Check boxes, add new findings.
6. **Commit and push** after completing each section.

---

## Starting Checklist (Phase 2)

```
[ ] Docker is running: docker compose -f BaratzaMemory/docker/docker-compose.yml up -d
[ ] PostgreSQL healthy: docker exec baratza-postgres pg_isready
[ ] Qdrant healthy: curl http://localhost:6333/healthz
[ ] NEOSY server running: cd BaratzaMemory && PYTHONPATH=src uvicorn baratza.api.app:app --port 8001
[ ] Create test_persistence.py in BaratzaMemory/tests/
[ ] Run: PYTHONPATH=src pytest tests/test_persistence.py -m integration -v
[ ] Create test_stress.py in BaratzaMemory/tests/
[ ] Run: PYTHONPATH=src pytest tests/test_stress.py -m integration -v
[ ] Update TESTING_TODO.md with results
[ ] Commit and push
```
