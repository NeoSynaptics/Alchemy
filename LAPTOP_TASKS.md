# Laptop Tasks — Single Window

Read this file on startup. Do tasks in order. Commit and push after each task. No GPU needed for any of these.

---

## Current State (2026-03-11 evening)

- Alchemy: 1109 passed, 8 failed (all 8 = missing deps, not code bugs)
- NEOSY: 98 passed, 0 failed
- Research tests: 63/70 — 3 known root causes, all fixable
- Serpentine scaffold exists at `testing/serpentine.py` — 7 stubs, needs flesh-out
- PC is handling GPU work independently via PC_TASKS.md

---

## Task 1 — Fix research test failures (already started)

```bash
pip install trafilatura
pytest tests/test_research/ -v
```

After trafilatura installs, 5 tests should auto-fix. Then fix the remaining 2:

**Fix 1:** `test_full_pipeline_success` in `tests/test_research/test_synthesizer.py`
- Problem: `time.perf_counter` mock returns 0.0 both calls, so `total_ms == 0`
- Fix: patch to return incrementing values:
  ```python
  with patch("time.perf_counter", side_effect=[0.0, 0.1]):
  ```
  Or relax assertion: `assert result.total_ms >= 0`

**Fix 2:** `test_search_sync` in `tests/test_research/` (DDGS v8 API change)
- Problem: test asserts `mock.assert_called_once_with(query, max_results=3)` but DDGS v8 now passes `region='wt-wt', safesearch='moderate'` by default
- Fix: update assertion to `mock.assert_called_once_with(query, max_results=3, region='wt-wt', safesearch='moderate')`
  OR use `mock.assert_called_once()` and check just `mock.call_args.kwargs['max_results'] == 3`

Re-run `pytest tests/test_research/ -v`. Target: 70/70.
Update `testing/TESTING_TODO.md` baselines table. Commit + push.

---

## Task 2 — Flesh out serpentine steps 1-5

Read `testing/serpentine.py`. The 7 steps are stubs. Fill in real HTTP calls for steps 1-5.

**Reference patterns** from `tests/integration/test_stress.py` (use same style).

- NEOSY runs on `http://localhost:8001`
- Alchemy runs on `http://localhost:8000`
- Always use `time.perf_counter()` for timing
- Always use `async with httpx.AsyncClient() as c:`

Steps to implement:
1. **Health check** — GET `http://localhost:8001/health` + GET `http://localhost:8000/health`
2. **Ingest 5 texts** — POST `http://localhost:8001/ingest` × 5 (pole vault topic items), collect memory_ids
3. **Search** — POST `http://localhost:8001/search` with `{"text": "pole vault approach speed"}`, assert ≥1 result
4. **Pin** — POST `http://localhost:8001/memories/{id}/pin` on first result, verify registro entry
5. **Batch 100** — POST `http://localhost:8001/ingest/batch` with 100 items, verify completed==100

Steps 6-7 (search latency under load, restart+verify) — leave as stubs with `# TODO: needs GPU/Docker`.

Commit + push.

---

## Task 3 — Synthetic voice test scaffold (no GPU needed to write)

Create `tests/test_voice/test_synthetic_voice.py`.

**The idea:** Use Piper TTS (CPU-only, already in `alchemy/voice/tts.py`) to generate WAV files from text, then feed through Whisper STT and verify round-trip transcript fidelity. Makes voice tests deterministic — no microphone needed.

First, read these files to understand the API:
- `alchemy/voice/tts.py` — find the TTS class and how to call it
- `alchemy/voice/stt.py` or `alchemy/voice/pipeline.py` — find Whisper STT API

Then write the test with these cases:
```python
COMMANDS = [
    "search for pole vault approach speed",
    "show settings",
    "ingest this page",
    "what did I save today",
    "search for paella recipe",
]
```

For each command:
1. TTS → WAV bytes (in-memory, no disk write needed)
2. WAV → Whisper STT → transcript
3. Assert word overlap between original and transcript ≥ 70%

If TTS or STT classes require the full server to be running, write the test but mark it `@pytest.mark.gpu` and add a note that it needs Alchemy server. Don't try to run it on the laptop.

Commit + push.

---

## Task 4 — Update TESTING_TODO baselines

After Tasks 1-3, update `testing/TESTING_TODO.md`:
- Baselines table: update Alchemy to reflect new research test pass count
- Section 6.3: add note "Synthetic voice test scaffold created — `tests/test_voice/test_synthetic_voice.py`"
- Section 9: mark serpentine steps 1-5 as scaffolded

Commit + push.

---

## Rules

- No GPU, no Docker required for any of these tasks
- Don't run integration tests — only unit tests and research tests
- Commit after every task
- If TTS/STT APIs are unclear, read the source first, don't guess
