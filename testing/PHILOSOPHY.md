# Testing Philosophy — Alchemy + BaratzaMemory

This document describes HOW and WHY we test. Any Claude window working on tests must read this first.

---

## Core Principle: Test Behavior, Not Plumbing

We don't care that an SQL string was generated. We care that **data injected into the system can be found again**. Every test should answer a user-visible question:

- "I saved something — can I find it?"
- "I dumped 10,000 photos — did any get lost?"
- "I restarted my computer — is my data still there?"
- "The GPU froze — did the voice still work?"

If a test doesn't answer a question a human would ask, it's not worth running.

---

## Synthetic Data Injection

Tests use **synthetic data** — realistic content that exercises the full pipeline. Not random strings. Not lorem ipsum. Content that looks like what the user actually stores:

- Pole vault training notes (sports + biomechanics)
- Instagram recipe saves (cooking + social media)
- YouTube lecture notes (education + math)
- Spanish-language science notes (multilingual)
- Code snippets with explanations (technical)
- NEO's own observations and pattern detections (AI entity)

The synthetic data generator lives in `BaratzaMemory/tests/synthetic.py`. It produces:
- **Realistic memories** — 10 curated examples across topics, entities, platforms, languages
- **Edge cases** — injection attacks, unicode extremes, size limits, duplicates, null bytes
- **Batch generators** — `make_batch(n)` for stress testing, `make_stress_batch(n)` for load testing

### Edge Cases Are Not Optional

Edge cases must **never silently fail**. The system either:
1. Handles it correctly (stores it, searches it, returns it), OR
2. Fails with a **clear, logged error** that a human or Claude can read and fix

Silent failure = the worst bug. A test that passes but silently dropped data is worse than a crash.

---

## The Size Ladder Pattern

For any operation that processes variable-size input (ingest, image classification, video transcription), we test with **increasing sizes** until the system breaks:

```
1MB → 10MB → 50MB → 100MB → 500MB → 1GB
```

Then we:
1. **Plot** size vs time to find the knee of the curve
2. **Set the limit** at the knee + 40% overhead margin
3. **Add config validation** that rejects inputs above the limit with a clear error
4. **Auto-resize** where possible (images before Qwen-VL, not rejection)

This pattern applies to: text ingest, image classification (Qwen-VL), video processing (ffmpeg + Whisper), batch sizes, and property JSONB depth.

---

## Buffer-First Architecture

The system is designed around a core insight: **ingest is fast and dumb, classification is slow and smart**.

When testing, this means:
- A massive data dump (10,000 photos from 10 phones) should **never block day tasks**
- The buffer accepts everything as fast as bandwidth allows
- Classification happens later, budgeted, prioritized
- User's real-time operations (search, ingest 1 file, voice commands) always get priority
- If the buffer grows huge, the system asks the user to cluster/prioritize — it doesn't freeze

**Stress test targets:**
- 10x normal load = must handle cleanly
- 20x normal load = should handle, acceptable degradation
- Beyond 20x = find the breaking point, set the limit

---

## Persistence Is Non-Negotiable

Data must survive:
- Docker restart
- Server process crash mid-ingest
- Computer shutdown and reboot

If data is in RAM and not on disk, it's a bug. Every test suite should include a "restart and verify" step. The append-only registro pattern means we can always trace what happened — if a registro entry exists, the operation happened. If it doesn't, it didn't.

Transaction safety: if the system crashes between writing to PostgreSQL and Qdrant, the memory row should exist with `status=RAW` (not `embedded`). No orphaned vectors without DB records. No DB records pointing to missing vault files.

---

## GPU Is a Shared Scarce Resource

The RTX 5060 Ti (16GB) and RTX 4070 (12GB) run everything: Qwen 14B, Whisper, embedding models, Qwen-VL, voice pipeline. VRAM management is life or death.

**Known pain points to test:**
- Small models (0.5-1GB) sitting idle as "broken links" consuming VRAM, causing Qwen to overflow to CPU or freeze entirely
- A 3-second inference becoming 30-60 seconds because of VRAM pressure
- Models silently flowing to CPU without the system knowing

**APU tests must verify:**
- VRAM accounting matches reality (nvidia-smi)
- Eviction actually frees VRAM (not just marks it free in a dict)
- Voice is NEVER evicted for batch work
- Frozen models are detected (timeout) and killed
- After cleanup, the correct models reload automatically

---

## Visual Debugging Is First-Class

Logs lie. Screenshots don't.

When testing UI or dashboard state, take a **real screenshot** and assert against it. The pattern:
1. Perform an action (load model, ingest data, trigger search)
2. Screenshot the dashboard/UI via Playwright
3. Assert visible state matches expected state (cards present, values correct, no error banners)
4. If assertion fails, save the screenshot as evidence

Example: APU debugger says 2GB free VRAM. Screenshot shows GPU card saying 2GB free. nvidia-smi says 2GB free. All three must agree. If they don't, the test fails and all three values are logged.

---

## NEO Is an Entity, Not Just Code

NEO is the AI half of the dual-entity system. It makes decisions: what to classify, what to skip, what connections to draw, what to tell the user. These decisions need testing too.

**Classification quality:** Inject a document about "pole vault biomechanics" and expect NEO to extract topics like `["sports", "biomechanics", "pole vault"]`. If it says `["technology"]`, that's a bug.

**Thought coherence:** When NEO generates a thought linking two memories, the connection should make semantic sense. Not random associations.

**Safety boundary:** NEO suggests external resources but NEVER auto-fetches from the web. This is a hard constraint that must be tested.

**Personality consistency:** NEO's tone and style should be consistent across interactions. If it's helpful in interaction 1 and dismissive in interaction 2, that's a behavioral bug.

These tests need curated expected outputs — they're not pass/fail assertions on code, they're quality evaluations of AI behavior.

---

## The Serpentine Test

The ultimate test is a single unattended script that walks through every major path:

Ingest → Search → Pin → Classify → Voice → Batch → Stress → Restart → Verify

Every step is timed. Every step logs its result. Any failure captures a full debug dump (screenshot + logs + DB state + VRAM state). This test should run nightly on the PC and push results to `testing/results/`.

---

## Multi-Language Support

The system supports English and Spanish natively (BGE-M3 is multilingual). In the future, models can think in any language internally — but user-facing output is in the user's preferred language.

For testing: inject content in Spanish, search in English, verify the result is found. The embedding model handles the bridge. If cross-language search doesn't work, it's an embedding model issue, not a code issue.

---

## Test Architecture

### Two Layers: Mocked (Laptop) + Real (PC)

**Laptop tests** run against mocked infrastructure — no Docker, no GPU. They verify logic, edge cases, and behavioral contracts. All 98 BaratzaMemory tests and 1100+ Alchemy tests are laptop-safe.

**PC tests** run against real Docker services (PostgreSQL, Qdrant) and real GPUs. They verify persistence, performance, VRAM management, and end-to-end flows. These are TESTING_TODO.md Sections 1-9.

The boundary is clean: if a test needs `docker compose up` or `nvidia-smi`, it's PC-only. Everything else runs anywhere.

### pytest Markers

```
@pytest.mark.integration  — requires Docker (PostgreSQL + Qdrant running)
@pytest.mark.benchmark    — size ladder / performance measurement
@pytest.mark.gpu          — requires GPU for model inference
@pytest.mark.serpentine   — the full end-to-end walk
```

Laptop: `pytest tests/` (unmarked = laptop-safe)
PC: `pytest tests/ -m integration` or `pytest tests/ -m gpu`

### The Timing Rule

Every PC test prints its timing:

```python
t0 = time.perf_counter()
# ... do the thing ...
elapsed = time.perf_counter() - t0
print(f"\n  {test_name}: {elapsed:.1f}s")
```

Without timing, we can't build size ladders, detect regressions, or set limits.

### The Three-Source Truth Pattern

For GPU state, three sources must agree:

1. **The code** — what the APU reports via API
2. **The screen** — what the dashboard shows (screenshot)
3. **The hardware** — what `nvidia-smi` reports

If any two disagree, the test fails and all three values are logged.

### Dev Order (4 Phases)

**Phase 1 — Laptop-safe:** Fix voice timing mocks, build test harness, scaffold Serpentine.
**Phase 2 — PC + Docker (no GPU):** Persistence & Recovery (Section 3), Buffer Stress (Section 2).
**Phase 3 — PC + GPU:** Size Ladders (1), APU Stress (4), Voice (6), NEO Intelligence (8).
**Phase 4 — Polish:** Playwright Scraping (5), Visual Debugging (7), wire into Serpentine, nightly runs.

See `PC_TEST_GUIDE.md` for full implementation specs per section.

---

## Where Tests Live

| Location | What |
|----------|------|
| `BaratzaMemory/tests/` | BaratzaMemory unit + behavioral + edge case tests (run on laptop, mocked infra) |
| `BaratzaMemory/tests/synthetic.py` | Synthetic data generator |
| `Alchemy/tests/` | Alchemy unit tests (1100+ tests) |
| `Alchemy/testing/TESTING_TODO.md` | Master checklist — all test sections with checkboxes |
| `Alchemy/testing/PHILOSOPHY.md` | This file — the WHY behind the tests |
| `Alchemy/testing/run_tests.sh` | Script to run both repos' tests |
| `Alchemy/testing/results/` | Raw test output (gitignored, regenerated on each run) |

## How to Work on Tests

### If You're on the Laptop (no GPU, no Docker):
1. Read this file (PHILOSOPHY.md)
2. Work on laptop-safe tests: logic, edge cases, behavioral contracts
3. Run: `cd BaratzaMemory && PYTHONPATH=src pytest tests/`
4. Run: `cd Alchemy && pytest tests/ -v`
5. Fix failures in the code, not the test
6. Commit and push

### If You're on the PC (GPU + Docker):
1. Read this file (PHILOSOPHY.md)
2. Read `PC_TEST_GUIDE.md` for implementation specs and dev order
3. Read TESTING_TODO.md for the checklist
4. Follow the phase order: Persistence → Stress → Size Ladders → APU → Voice → NEO
5. Every test prints timing. Every failure captures a debug dump.
6. Update TESTING_TODO.md with results (check boxes, add findings)
7. Commit and push
