# Testing TODO

Distilled from test results + user requirements. Claude reads this, fixes issues, re-runs tests.

**Implementation order:** Phase 1 (laptop) → Phase 2 (Docker) → Phase 3 (GPU) → Phase 4 (polish).
See `PC_TEST_GUIDE.md` for full implementation specs with code examples.

---

## STATUS (2026-03-11)

- **Phase 1: DONE** — timing mocks fixed, serpentine scaffold, baselines
- **Phase 2: DONE** — persistence, stress, 10K batch FIXED (chunked ingest, 29.6s)
- **Phase 3: IN PROGRESS** — APU baseline 8/8, VRAM leak 3/3, concurrency/priority scaffolded
- **Phase 4: NOT STARTED**

### Blocking Issues
1. ~~**10K batch: 11/10000 failures**~~ — **FIXED.** Chunked batch ingest (29.6s, 3ms/item).
2. **APU VRAM drift 5-8GB** — `_make_room()` trusts registry, not nvidia-smi. OOM risk on tight GPUs. Fix: pre-load reality check. See Alchemy TODO Task A.

### Priority Order
1. **PC**: APU VRAM reality check → run live GPU tests → concurrency/priority tests
2. **Laptop**: Research test fixes → baselines update
3. **Defer**: Serpentine flesh-out, synthetic voice, Section 6 voice reliability

### Test Suite Health Review (2026-03-11)

**STOP expanding synthetic/stress tests. Start consolidating.**

14K LOC tests vs 26K LOC production (1.84:1). APU alone: 3,200 LOC tests for 2,600 LOC source (120%).

**Consolidation tasks (do before writing ANY new tests):**
1. Extract shared GPU safety helpers to `tests/test_apu/conftest.py`
2. Merge duplicate voice latency tests into one location (priority_live owns it)
3. Merge test_vram_live.py into test_apu_concurrency_live.py (or delete if fully redundant)
4. Deepen test_apu_api.py: error cases, response content validation, edge inputs
5. Add invariant checks AFTER every operation in test_apu_stress.py

---

## Current Baselines (2026-03-11)

| Repo | Passed | Failed | Notes |
|------|--------|--------|-------|
| Alchemy | 1128 | 0 | 13 modules, 39 deselected (GPU-only) |
| Alchemy GPU (APU) | 8 | 0 | VRAM accounting, model tracking, voice, perf |
| NEOSY | 98 | 0 | 42 unit + 14 behavioral + 42 edge cases |
| NEOSY integration | 21 | 0 | 7 persistence + 9 stress + 5 image ladder. 10K batch FIXED |
| NEOSY benchmark | 10 | 0 | Size ladder + batch ladder + search perf |

### Known Failures
1. ~~**5 timing mock issues** — FIXED~~
2. **8 missing `duckduckgo_search` dep** — `test_research/` needs the package installed (PC only)

### Remaining Action Items
- [ ] Install `duckduckgo_search` on PC and re-run research tests
- [ ] Run NEOSY edge cases against real Docker + Qdrant on PC

---

## Pending Work — Phase 3 (GPU)

### 1.1 Ingest Size Ladder (upper bounds)
- [x] 1MB–100MB measured. Knee of curve ~50MB.
- [ ] 500MB → measure (target upper bound)
- [ ] 1GB → find where it breaks
- [ ] Set MAX_INGEST_SIZE = ~70MB + config validation

### 1.2 Image Size Ladder — DONE
- [x] 1MP–50MP all pass. Cold start 7.31s, steady ~1.3s. 50MP did NOT break.
- [ ] Set MAX_IMAGE_RESOLUTION, auto-resize before Qwen-VL

### 1.3 Video Processing Ladder
- [ ] 30s video (720p) → ffmpeg + whisper time
- [ ] 5min video (1080p) → time
- [ ] 30min video → time
- [ ] 2hr video → time
- [ ] Plot: duration vs processing time

### 2.3 Priority & Day Tasks (remaining)
- [ ] 5,000 items in background queue + user ingests 1 item → classified within 30s
- [ ] NEO classification of bulk does NOT block user operations
- [ ] If buffer gets huge (20,000 photos), system asks user to cluster/prioritize

### 4.1-4.2 VRAM Live Tests — 3/3 PASS
- [x] Load/unload small model: PASS (0.5s load, 0.4s unload, -13MB leak)
- [x] 10 load/unload cycles: PASS (-14MB total leak, drift stable at 271MB)
- [x] APU health + event tracking: PASS (500ms, 100 events, 0 slow)
- [ ] Small model idle 5min → APU reclaims if Qwen needs VRAM
- [ ] Qwen needs 10GB, only 8GB free → small model MUST be evicted
- [ ] Model takes 60s instead of 3s → APU logs "slow"
- [ ] Model never returns (infinite hang) → timeout fires, model killed
- [ ] After timeout kill → GPU VRAM actually freed
- [ ] Voice pipeline stays alive during GPU crash/recovery

### 4.3 APU Priority
- [ ] Voice + Qwen + embeddings loaded → voice never evicted for batch work
- [ ] Voice command during Qwen classification → Qwen yields, voice responds <2s
- [ ] Batch classification running → user ingest → embedding model loads within 5s

### 4.4 Synthetic APU Chaos
- [ ] Load wrong model intentionally → APU detects, cleans up, logs error
- [ ] Fill GPU with synthetic waste → APU cleans and restarts correct models
- [ ] Voice keeps running during GPU cleanup

### 4.5 GPU Concurrency & Multi-App Contention
**Test scaffolds ready:** `tests/test_apu/test_apu_concurrency_live.py` + `test_apu_priority_live.py`

**Concurrency:**
- [ ] 2 concurrent loads of SAME model → no double VRAM allocation
- [ ] Load during unload race → model ends in valid state
- [ ] 3 concurrent load requests → VRAM accounting stays valid
- [ ] Voice responds during concurrent model loads
- [ ] Voice responds during rapid load/unload churn (3 cycles)
- [ ] APU status returns valid data during model operations
- [ ] Event log captures all concurrent operations
- [ ] nvidia-smi matches APU tracking after load/unload cycle

**Priority:**
- [ ] Default priorities correct: voice=10 (highest), gate=60 (lowest)
- [ ] All GPU models report valid tier + location
- [ ] High-priority app activation always succeeds
- [ ] Low-priority app model demoted after deactivation
- [ ] Voice latency baseline <2s with no GPU pressure
- [ ] Voice latency <2s during model load
- [ ] APU health check detects and reports VRAM drift
- [ ] Every operation generates event log entry
- [ ] Frozen baseline restore returns server to known-good state

**Known APU limitations (documented in orchestrator.py TODO):**
- Global `asyncio.Lock()` serializes ALL model ops across both GPUs
- No load queue prioritization — FCFS when waiting for lock
- No preemption of in-flight loads
- No request deduplication for same-model concurrent loads

---

## Pending Work — Phase 4 (Polish)

### 5. Playwright & Scraping
- [ ] Instagram saves scrape → data arrives (not empty)
- [ ] Login expired → clear error
- [ ] Page structure changed → error with context
- [ ] Scrape → ingest → search pipeline (text, images, video)

### 6. Voice Reliability
- [ ] Command reliability: 20 reps each of top 10 commands, target 100%
- [ ] Voice under GPU load: responds <3s during classification/embedding
- [ ] Synthetic voice testing: TTS → STT → full pipeline

### 7. Visual Debugging & Screenshot Assertions
- [ ] Dashboard: APU cards visible, model table, event feed
- [ ] Settings: toggles present, changes reflected
- [ ] NEO Activity Debugger: timeline, decisions, stuck detection
- [ ] APU Visual Debugger: VRAM per GPU, real-time events

### 8. NEO Personality & Intelligence
- [ ] Classification quality: known topics, language detection
- [ ] Thought coherence: semantic links, no hallucination
- [ ] Cross-language search: BGE-M3, typo tolerance
- [ ] NEO as entity: consistent tone, correction handling, no auto-fetch

### 9. End-to-End Serpentine
- [ ] Single unattended test through all major paths (15 steps)
- [ ] All steps logged with timing
- [ ] Failure at any step = full debug capture

---

## Completed Work (reference)

<details>
<summary>Phase 1 — Laptop (DONE)</summary>

- Voice timing mocks fixed (279 voice tests passing)
- Serpentine scaffold created
</details>

<details>
<summary>Phase 2 — PC + Docker (DONE)</summary>

- Mass ingest stress: 100/1K/10K items all pass
- Concurrent multi-device: 20 streams × 10 items, 0 errors
- 10K batch: FIXED via chunked ingest (29.6s, 3ms/item)
- Persistence: restart survival, transaction safety all pass
- Ingest size ladder: 1MB–100MB measured, knee at ~50MB
- Batch size ladder: 10–2000 items, linear scaling
- Search baseline: avg 84ms, worst 306ms cold
</details>

<details>
<summary>Phase 3 — Completed items</summary>

- APU baseline 8/8 PASS (VRAM, fleet, GPUs, voice, latency, modules)
- VRAM leak detection 3/3 PASS (load/unload, 10 cycles, health)
- Image size ladder: 1MP–50MP all pass, steady ~1.3s
</details>

---

## How to Update This File

1. Run `bash testing/run_tests.sh`
2. Read results in `testing/results/`
3. Update this file with new failures/fixes
4. Commit and push
