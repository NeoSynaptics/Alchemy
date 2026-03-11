# PC Tasks — Single Window

Read this file on startup. Do tasks in order. Commit and push after each task. Do NOT run multiple tests at once — GPU is shared with Ollama/Qwen.

---

## Current State (2026-03-11 evening)

- Alchemy server: running on :8000, Qwen loaded on 5060 Ti (~14.8GB)
- APU baseline 8/8: confirmed passing (57ms avg, 266ms worst)
- Phase 2 (NEOSY Docker): DONE
- Phase 3 (GPU): IN PROGRESS — APU baseline done, VRAM live tests next

---

## Task 1 — VRAM health check (read-only, safe)

Run this single test first to confirm VRAM scaffold is wired:

```bash
pytest tests/test_apu/test_vram_live.py::TestFrozenRoutine::test_apu_detects_slow_inference -v -s -m gpu
```

This only reads APU status and health endpoints — no model loads, no GPU pressure.

**Expected:** 1 test passes, prints APU event count + health status.

Update `testing/TESTING_TODO.md` Section 4.2 with result. Commit + push.

---

## Task 2 — VRAM load/unload cycle (touches GPU — do alone)

Run ONE load/unload test:

```bash
pytest tests/test_apu/test_vram_live.py::TestVRAMLeak::test_load_unload_small_model_vram_returns -v -s -m gpu
```

**What it does:** Loads `qwen2.5:0.5b`, unloads it, checks VRAM returns within 200MB of baseline.

**If `/v1/apu/load` returns 404:** Run `curl http://localhost:8000/docs 2>/dev/null | grep -i "apu\|load\|unload"` to find the real endpoint names. Report back what endpoints exist — do NOT modify the test yet, just report.

**If it passes:** Update TESTING_TODO.md Section 4.1, commit + push. Then move to Task 3.

**If it fails or hangs >2 min:** Stop, report the error. Do not retry.

---

## Task 3 — 10 cycle VRAM leak check

Only run this if Task 2 passed:

```bash
pytest tests/test_apu/test_vram_live.py::TestVRAMLeak::test_10_load_unload_cycles_no_drift -v -s -m gpu
```

**What it does:** 10 load/unload cycles of the small model, checks total VRAM drift <500MB.

Update TESTING_TODO.md Section 4.1 with result. Commit + push.

---

## Task 4 — NEOSY 10K batch fix (no GPU needed)

Switch to BaratzaMemory repo:

```bash
cd ~/BaratzaMemory  # adjust path if different
git pull
```

Read `src/baratza/ingest/pipeline.py` — find where batch items are processed concurrently (look for `asyncio.gather` or a loop over items). Add a semaphore to cap concurrent DB writes:

```python
_batch_semaphore = asyncio.Semaphore(8)

async def _ingest_one(item):
    async with _batch_semaphore:
        # existing per-item ingest logic
        ...
```

Then re-run:

```bash
pytest tests/integration/test_stress.py::TestMassIngestExtended::test_10000_batch_no_loss -v -s
```

Target: 10000/10000. If it passes, update TESTING_TODO.md Section 2.1 (change "11 failed" to "0 failed"), commit + push.

---

## Task 5 — Image size ladder (GPU, run alone)

Only after Tasks 1-4 are done and GPU is idle.

```bash
cd ~/BaratzaMemory  # adjust path
pip install Pillow
pytest tests/integration/test_image_ladder.py -v -s
```

This runs 1MP → 50MP image ingests. Does NOT hard-fail — records where Qwen-VL starts struggling. Each image may take 30-90s. Total: ~10 min.

Record each result in TESTING_TODO.md Section 1.2. Commit + push.

---

## Task 6 — GPU Concurrency & Priority tests (GPU, run alone)

Only after Tasks 1-5 are done. These test what happens when multiple things hit the GPU simultaneously.

**IMPORTANT SAFETY:** These tests have built-in safety nets — they bail out if VRAM is too low, use only small models (0.5b), and auto-cleanup after each test with frozen baseline restore. Still, run them one at a time.

First, run the concurrency tests:

```bash
cd ~/Documents/Alchemy_explore
pytest tests/test_apu/test_apu_concurrency_live.py -v -s -m gpu
```

**What it tests:** Concurrent model loads (double allocation prevention), load/unload race conditions, voice survival during GPU churn, status consistency during operations.

**Expected:** All tests pass or skip (SAFETY BAIL if VRAM too low). If any test hangs >2 min, kill it.

Then run the priority tests:

```bash
pytest tests/test_apu/test_apu_priority_live.py -v -s -m gpu
```

**What it tests:** Priority ordering (voice=10 highest), app activation/deactivation, voice latency during model loads, event logging completeness, frozen baseline restore.

Update TESTING_TODO.md Section 4.5 with results. Commit + push.

---

## Rules

- One test at a time
- Never run two GPU-heavy tests simultaneously
- If anything hangs >3 min, kill it and report
- Commit after every task, don't batch commits
- Update TESTING_TODO.md with real numbers after every test
