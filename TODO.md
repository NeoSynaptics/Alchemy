# Alchemy — Current Tasks

Read this file when you start. Do tasks in order, top to bottom. Skip tasks marked [DONE]. Commit and push after each task.

**Repo:** `C:\Users\info\GitHub\Alchemy` (branch: main)

---

## ═══════════════════════════════════════════
## ACTIVE — Do These Next
## ═══════════════════════════════════════════

## [DONE] Task A: APU VRAM Reality Check — Prevent OOM on Tight GPUs

Implemented: `_make_room()` and `_auto_select_gpu()` now use `min(nvidia-smi, registry)` for VRAM estimation. Pre-load reality check with 200MB safety margin, post-load drift detection (>1GB logged), OOM recovery (async reconcile after failure). Config: `settings.apu.vram_safety_margin_mb`. All 110 APU tests pass.

---

## Task B: Test Suite Consolidation (before running any new live tests)

14K LOC tests vs 26K LOC production. APU at 120% test density. **Stop expanding, start consolidating.**

1. Create `tests/test_apu/conftest.py` — extract shared helpers:
   - `get_real_vram()` (duplicated 4x with different implementations)
   - `assert_vram_safe()` (duplicated 2x)
   - `vram_snapshot_str()` (duplicated 2x)
2. Merge voice latency tests — currently in BOTH priority_live AND concurrency_live → priority_live owns voice latency, remove from concurrency_live
3. Merge or delete test_vram_live.py — overlaps heavily with test_apu_concurrency_live.py
4. Deepen test_apu_api.py — currently only checks status codes. Add error cases, response content validation
5. Add `await assert_apu_invariants(orch)` after EVERY operation in test_apu_stress.py

**Do NOT write new test files until these 5 items are done.**

**Files:** `tests/test_apu/conftest.py` (new), `tests/test_apu/test_apu_priority_live.py`, `tests/test_apu/test_apu_concurrency_live.py`, `tests/test_apu/test_vram_live.py`, `tests/test_apu/test_apu_api.py`, `tests/test_apu/test_apu_stress.py`
**Commit when done.**

---

## Task C: Research Test Fixes (laptop, no GPU)

8 tests fail due to missing deps + 2 mock issues.

1. `pip install trafilatura` — fixes 5 tests
2. Fix `test_full_pipeline_success` in `tests/test_research/test_synthesizer.py` — perf_counter mock returns 0.0 both times → patch with `side_effect=[0.0, 0.1]`
3. Fix `test_search_sync` — DDGS v8 API change, use `mock.assert_called_once()` + check `call_args.kwargs['max_results'] == 3`

Target: 1117 passed, 0 failed. **Commit when done.**

---

## Task D: Run GPU Live Tests — Prove APU Survives Double-Hits (PC only)

Only after Task A (VRAM safety) and Task B (consolidation) are done.

**Why this matters:** In production, voice + gate + click will all hit the APU at the same time. Two apps requesting the same model simultaneously, a load racing an unload, three concurrent loads competing for the last 2GB of VRAM. Mocks can't catch real OOM — only real GPU tests can.

```bash
pytest tests/test_apu/test_apu_concurrency_live.py -v -s -m gpu
pytest tests/test_apu/test_apu_priority_live.py -v -s -m gpu
```

**Critical scenarios that MUST pass:**
1. Two concurrent loads of SAME model → no double VRAM allocation, no OOM
2. Load-during-unload race → model ends in valid state (not corrupted)
3. 3 concurrent loads competing for VRAM → accounting stays valid, no overcommit past physical GPU
4. Voice responds <2s during concurrent model loads (voice must NEVER starve)
5. nvidia-smi matches APU tracking after the storm settles

**If tests fail:** Fix the orchestrator, not the tests. These are real race conditions.

Check off items in `testing/TESTING_TODO.md` Section 4.5 as each passes. **Commit fixes after each bug found.**

---

## ═══════════════════════════════════════════
## BACKLOG — Do Later
## ═══════════════════════════════════════════

## Task E: NEOSY Bug Fixes (port, CLI column, vault path)

Three quick fixes found during code review.

1. `.env.example` says `API_PORT=8000` but `config.py` defaults to 8001 → fix `.env.example`
2. `cli.py` accesses `c["connection_type"]` but DB column is `connection` → fix accessor
3. `vault_path: Path = Path("./vault")` is relative → resolve to absolute

**Repo:** BaratzaMemory. **Commit when done.**

---

## Task F: Voice Reliability Testing (Phase 3, needs GPU)

- Synthetic voice scaffold: TTS → WAV → STT round-trip
- Command reliability: 20 reps of top 5 commands
- Voice under GPU load: latency during Qwen classification

See `testing/TESTING_TODO.md` Section 6 for details.

---

## Task G: Serpentine E2E (Phase 4, needs everything running)

See `testing/TESTING_TODO.md` Section 9. Full unattended integration test.
Only after all Phase 3 items are done.

---

## ═══════════════════════════════════════════
## COMPLETED (28 tasks)
## ═══════════════════════════════════════════

**Tasks 1-4:** GPU fleet registration, NEOSY settings, server mount, APU stabilization
**Tasks 5-8:** APU event logger, concurrency fix (atomic decide-evict-load), stress tests + invariants, health endpoint + periodic reconciliation
**Tasks 9-10:** APU concurrency review (5 bugs fixed), audit fixes (rollback tier, stateful FakeGPUMonitor)
**Task 11:** Portability pass (hardcoded paths removed, .env.example created)
**Tasks 12a-d:** Orchestrator race conditions, registry thread-safety, event log filter fix, invariant checker hardening, test quality improvements
**Tasks 15-22:** Dashboard wiring, security middleware, AlchemyWord, Docker, AlchemyHole spec, voice audit, import boundaries, README
**Tasks 23-27:** Dashboard E2E, NEOSY E2E with Ollama, voice pipeline E2E, RLHF foundation, full integration smoke test
