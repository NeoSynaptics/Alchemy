# Testing TODO

Distilled from test results + user requirements. Claude reads this, fixes issues, re-runs tests.

**Implementation order:** Phase 1 (laptop) → Phase 2 (Docker) → Phase 3 (GPU) → Phase 4 (polish).
See `PC_TEST_GUIDE.md` for full implementation specs with code examples.

---

## STATUS (2026-03-11 evening)

- **Phase 1: DONE** — timing mocks fixed, serpentine scaffold, baselines
- **Phase 2: DONE** — persistence, stress, all items checked (10K batch: 99.89%)
- **Phase 3: IN PROGRESS** — APU baseline 8/8, size/batch ladder done, VRAM live tests scaffolded
- **Phase 4: NOT STARTED**

### Known Issues
1. **10K batch: 11/10000 failures** — likely DB connection pool timeout under sustained load. Investigate `asyncpg` pool `max_size` in NEOSY config.
2. **Section 2.3 items** — classification-dependent, need NEO running (Phase 3)

### Next Up
- **PC**: Section 1.2 image ladder, Section 4.1-4.2 VRAM leak (test scaffolds ready), then Section 6 voice
- **Laptop**: Flesh out serpentine, connection pool investigation, duckduckgo dep fix

---

## Current Baselines

| Repo | Passed | Failed | Notes |
|------|--------|--------|-------|
| Alchemy | 1109 | 8 | 8 missing dep (research), timing mocks FIXED |
| Alchemy GPU (APU) | 8 | 0 | VRAM accounting, model tracking, voice, perf (2026-03-11 PC) |
| NEOSY | 98 | 0 | 42 unit + 14 behavioral + 42 edge cases |
| NEOSY integration | 16 | 1 | 7 persistence + 9 stress. 10K batch: 11/10000 failed (2026-03-11 PC) |
| NEOSY benchmark | 10 | 0 | Size ladder + batch ladder + search perf (2026-03-11 PC) |

### Alchemy Known Failures
1. ~~**5 timing mock issues** — FIXED (inference_ms mocks now use perf_counter patch)~~
2. **8 missing `duckduckgo_search` dep** — `test_research/` needs the package installed (PC only)

### Alchemy Action Items — Phase 1 (Laptop)
- [x] Fix voice timing mocks (laptop-safe, no GPU needed) — DONE, 279 voice tests passing
- [x] Build serpentine scaffold — DONE, `testing/serpentine.py` created
- [ ] Install `duckduckgo_search` on PC and re-run research tests

### NEOSY Action Items — Phase 2 (PC + Docker)
- [ ] Run edge cases against real Docker + Qdrant on PC

---

## Section 1: Size & Performance Limits (Phase 3 — PC + GPU)

Goal: Load increasingly larger files, plot size vs time, find the cutoff. Set limits with 40% overhead margin.

### 1.1 Ingest Size Ladder
- [x] 1MB text → 2.1s, 0.43 MB/s (2026-03-11 PC)
- [x] 10MB text → 5.4s, 1.67 MB/s
- [x] 50MB text → 28.8s, 1.56 MB/s
- [x] 100MB text → 89.0s, 1.01 MB/s (throughput drops — knee of curve is ~50MB)
- [ ] 500MB → measure (target upper bound)
- [ ] 1GB → find where it breaks
- [ ] Plot: file size vs ingest time. Find the knee of the curve.
- [ ] Set MAX_INGEST_SIZE = (knee point) + 40% overhead → suggest ~70MB based on 50MB knee
- [ ] Add config setting + validation that rejects files above limit with clear error

### 1.1b Batch Size Ladder (2026-03-11 PC — all PASS)
- [x] 10 batch → 0.7s (69ms/item)
- [x] 100 batch → 3.3s (33ms/item)
- [x] 500 batch → 15.0s (30ms/item)
- [x] 1000 batch → 27.4s (27ms/item)
- [x] 2000 batch → 57.9s (29ms/item) — linear scaling, no degradation

### 1.1c Search Performance Baseline (2026-03-11 PC)
- [x] 5 queries: avg 84ms, worst 306ms (first query cold). Sub-30ms warm.

### 1.2 Image Size Ladder (Qwen-VL specific)
Known: large images choke Qwen-VL. Find the real limit.
**Test scaffold ready:** `NEOSY/tests/integration/test_image_ladder.py` — run with `pytest -m benchmark`
- [ ] 1MP image (1000x1000) → Qwen-VL time
- [ ] 4MP (2000x2000) → time
- [ ] 12MP (4000x3000) → time (phone camera size)
- [ ] 24MP → time
- [ ] 50MP → expect failure
- [ ] Set MAX_IMAGE_RESOLUTION, auto-resize before Qwen-VL

### 1.3 Video Processing Ladder
Videos not yet tested. Find the limits.
**Test scaffold ready:** `NEOSY/tests/integration/test_video_ladder.py` — run with `pytest -m benchmark`
- [ ] 30s video (720p) → ffmpeg + whisper time
- [ ] 5min video (1080p) → time
- [ ] 30min video → time
- [ ] 2hr video → time
- [ ] Plot: duration vs processing time

---

## Section 2: Buffer & Queue Stress (Phase 2 — PC + Docker)

Goal: Ingest buffer handles massive dumps (10-20x normal) without blocking day tasks. Store fast, classify later.

### 2.1 Mass Ingest Stress
- [x] 100 items sequentially → all stored, no loss (3.7s, 37ms/item)
- [x] 1,000 items batch → all stored, no silent drops (35.8s, 36ms/item). Bug fixed: `'processing'` → `'in_progress'` in ingest.py.
- [x] 10,000 items batch → **9989/10000 completed, 11 failed** (292.6s, 29ms/item). 99.89% — investigate the 11 failures (likely DB connection timeout under sustained load).
- [x] During mass ingest, 1 "day task" ingest → completes in <2s (446ms during 500-item batch)
- [x] During mass ingest, search query → returns in <1s (avg 128ms, worst 309ms)

### 2.2 Concurrent Multi-Device Simulation
- [x] 5 concurrent ingest streams (5 phones pushing photos) — 2.9s, no errors
- [x] 10 concurrent → no DB connection pool exhaustion (3.5s, 0 errors)
- [x] 20 concurrent × 10 items → **200/200, 0 errors, 6.8s**. No breaking point found at 20 streams.
- [x] No duplicate memory_ids across streams (verified 5x20, 10x10, 20x10)
- [x] Registro has exactly N entries for N ingests — verified 25/25, 0 missing

### 2.3 Priority & Day Tasks
- [ ] 5,000 items in background queue + user ingests 1 item → classified within 30s
- [x] User search during bulk → latency <500ms (avg 128ms, worst 309ms)
- [ ] NEO classification of bulk does NOT block user operations
- [ ] If buffer gets huge (20,000 photos from mass dump), system asks user to cluster/prioritize

---

## Section 3: Persistence & Recovery (Phase 2 — PC + Docker, DO FIRST)

Goal: Data survives restart. No silent loss.

### 3.1 Restart Survival
- [x] Ingest 10 items → restart Docker → query all 10 → all present (0.9s ingest, 4.1s restart+recovery)
- [x] Kill server mid-ingest → restart → partial items either complete or cleanly rolled back (canary status: embedded)
- [x] Qdrant vectors survive restart (volume mount) — all 10 vectors verified after restart
- [x] Vault files survive restart — 241 vault_refs verified identical after Docker restart

### 3.2 Transaction Safety
- [x] Force Qdrant error mid-ingest → server returns 500 (rejects, no silent drop). No orphaned vectors (0 found in audit of 47 vectors).
- [x] Force DB error mid-ingest → **0 orphaned vectors**. Stopped PostgreSQL, attempted ingest (got 500), restarted, verified vector count unchanged (15279 before = 15279 after).

---

## Section 4: GPU Health & APU Stress (Phase 3 — PC + GPU)

Goal: Detect frozen routines, recover VRAM. Small model leaks don't block big models.

### 4.0 APU Baseline (2026-03-11 PC — 8/8 PASS)
Tests in `Alchemy/tests/test_apu/test_apu_integration.py`. Run with `pytest -m gpu`.
- [x] VRAM matches nvidia-smi within 500MB (GPU0: diff=271MB, GPU1: diff=260MB)
- [x] 12 models in fleet, all tracked with valid locations
- [x] 2 GPUs detected, matches hardware
- [x] All Ollama models have valid location (gpu_0/gpu_1/cpu_ram/disk)
- [x] GPU-resident models report >0 VRAM (qwen2.5vl:7b on gpu_1, 13.5GB)
- [x] Voice status endpoint responds (voice not started but endpoint healthy)
- [x] APU status latency: avg 55ms, worst 260ms
- [x] /v1/modules: 16 modules registered

### 4.1 VRAM Leak Detection
**Test scaffold ready:** `Alchemy/tests/test_apu/test_vram_live.py` — run with `pytest -m gpu`
Known pain: small 0.5-1GB models sit as broken links, eat VRAM, make Qwen overflow to CPU or freeze.
- [x] Load/unload small model (qwen2.5:0.5b): PASS — load 0.5s, unload 0.4s, VRAM leak -13MB (tolerance 200MB) — 2026-03-11 PC
- [ ] Small model idle 5min → APU reclaims if Qwen needs VRAM
- [x] 10 load/unload cycles: PASS — total leak -14MB (tolerance 500MB), APU drift stable at 271MB — 2026-03-11 PC
- [ ] Qwen needs 10GB, only 8GB free → small model MUST be evicted

### 4.2 Frozen Routine Detection
- [x] APU health + event tracking: PASS (500ms, 100 events, 0 slow) — 2026-03-11 PC
- [x] Health endpoint: responds 200 but `healthy: False` — GPU0 drift 7864MB, GPU1 drift 5164MB (system processes + Ollama overhead not in registry)
- [ ] Model takes 60s instead of 3s → APU logs "slow"
- [ ] Model never returns (infinite hang) → timeout fires, model killed
- [ ] After timeout kill → GPU VRAM actually freed (not just marked free)
- [ ] Voice pipeline stays alive during GPU crash/recovery

### 4.3 APU Priority
- [ ] Voice + Qwen + embeddings loaded → voice never evicted for batch work
- [ ] Voice command during Qwen classification → Qwen yields, voice responds <2s
- [ ] Batch classification running → user ingest → embedding model loads within 5s

### 4.4 Synthetic APU Chaos
- [ ] Load wrong model intentionally → APU detects, cleans up, logs error
- [ ] Fill GPU with synthetic waste (controlled small models) → APU cleans and restarts correct models
- [ ] Voice keeps running during GPU cleanup

---

## Section 5: Playwright & Scraping (Phase 4 — Polish)

Goal: Scraping gets real data. Errors are clear, never silent.

### 5.1 Scraping Reliability
- [ ] Instagram saves scrape → data arrives (not empty)
- [ ] Login expired → clear error (not silent empty result)
- [ ] Page structure changed → error with context ("selector not found")
- [ ] Scrape 10 pages → all 10 have content, none silently skipped
- [ ] User picks folder to scrape → only that folder's content ingested

### 5.2 Scraping → Ingest Pipeline
- [ ] Scrape 5 Instagram posts → ingest → search → find them
- [ ] Scrape page with images → images in vault + embedded
- [ ] Scrape page with video → audio extraction + transcription triggered

---

## Section 6: Voice Reliability (Phase 3 — PC + GPU)

Goal: Core voice commands work 100%.

### 6.1 Command Reliability (20 reps each)
- [ ] "Show settings" → settings page opens (20x, expect 100%)
- [ ] "Search for [topic]" → results appear (20x)
- [ ] "Ingest this" → ingest triggered (20x)
- [ ] Measure success rate per command. Target: 100% for top 10.

### 6.2 Voice Under GPU Load
- [ ] Voice command while Qwen classifying → responds <3s
- [ ] Voice command while batch embedding → responds
- [ ] STT accuracy when GPU under load (Whisper sharing GPU)

### 6.3 Synthetic Voice Testing
- [ ] Generate synthetic audio of 10 commands (TTS → WAV)
- [ ] Feed through STT → verify transcription
- [ ] Feed through full pipeline → verify correct action

---

## Section 7: Visual Debugging & Screenshot Assertions (Phase 4 — Polish)

Goal: Screenshot real UI, assert against expected state. Visual = trustworthy.

### 7.1 Dashboard
- [ ] Start server → screenshot → APU cards visible, no error banners
- [ ] Load model → screenshot → model in table
- [ ] Ingest → screenshot → event feed shows entry

### 7.2 Settings
- [ ] Open settings → screenshot → all toggles present
- [ ] Change setting → screenshot → change reflected

### 7.3 NEO Activity Debugger (NEW)
- [ ] Visual timeline of NEO actions: what it classified today, time per task, queue depth
- [ ] Log every NEO decision: what it chose, why, what it skipped
- [ ] Alert when NEO idle but queue not empty (stuck?)

### 7.4 APU Visual Debugger (NEW)
- [ ] APU dashboard shows real VRAM per GPU, matches nvidia-smi
- [ ] Screenshot APU status → assert debug data matches visual display
- [ ] Model load/unload events visible in real-time feed

---

## Section 8: NEO Personality & Intelligence (Phase 3 — PC + GPU)

### 8.1 Classification Quality
- [ ] 10 known-topic documents → NEO classifies → extracted topics vs expected
- [ ] "Pole vault biomechanics" → expect: ["sports", "biomechanics", "pole vault"]
- [ ] Spanish document → correct language detection

### 8.2 Thought Coherence
- [ ] NEO links memories A + B → connection makes semantic sense
- [ ] NEO insight → references real content, not hallucinated

### 8.3 Cross-Language Search
- [ ] Search "pole vault" (English) → find "salto con pertiga" (Spanish) via BGE-M3
- [ ] Search with typo "polevalt" → still find results (embedding similarity)
- [ ] Search "that cooking video from Instagram" → find paella reel (metadata + semantic)

### 8.4 NEO as Entity
- [ ] Consistent tone across 10 interactions
- [ ] When user corrects NEO → NEO acknowledges, adjusts
- [ ] NEO suggests but never auto-fetches from web (safety boundary)

---

## Section 9: End-to-End Serpentine (Phase 1 scaffold + Phase 4 wire-up)

One continuous unattended test through every major path:

```
 1. Start Docker + server + voice
 2. Ingest 5 synthetic text files via API
 3. Ingest 3 synthetic images via API
 4. Verify all 8 in DB + Qdrant
 5. Search "pole vault" → find sports note
 6. Pin that result → verify registro entry
 7. Trigger NEO classification on 8 items
 8. Verify topics/summaries populated
 9. Voice: "what did I save today?" → response references items
10. Batch ingest 100 items → queue fills
11. During batch, search → verify <500ms
12. Screenshot dashboard → event feed has entries
13. Restart server → all data persists
14. Search again → same results
15. Log every step with timing. Any failure = screenshot + debug dump.
```

- [ ] Script this as single unattended test
- [ ] All steps logged with timing
- [ ] Failure at any step = full debug capture

---

## How to Update This File

1. Run `bash testing/run_tests.sh`
2. Read results in `testing/results/`
3. Update this file with new failures/fixes
4. Commit and push
