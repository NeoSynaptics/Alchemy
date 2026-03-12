# Laptop Tasks — Single Window

Read this file on startup. Do tasks in order. Commit and push after each task. No GPU needed for any of these.

---

## Current State (2026-03-11)

- Alchemy: 1109 passed, 8 failed (all 8 = missing deps, not code bugs)
- BaratzaMemory: 98 passed, 0 failed
- PC is handling GPU work via PC_TASKS.md
- **Priority: fix broken tests + consolidation, not new scaffolds**

---

## Task 1 — Fix research test failures

```bash
pip install trafilatura
pytest tests/test_research/ -v
```

After trafilatura installs, 5 tests should auto-fix. Then fix the remaining 2:

**Fix 1:** `test_full_pipeline_success` in `tests/test_research/test_synthesizer.py`
- `time.perf_counter` mock returns 0.0 both calls, so `total_ms == 0`
- Fix: `with patch("time.perf_counter", side_effect=[0.0, 0.1]):`

**Fix 2:** `test_search_sync` — DDGS v8 API change
- Use `mock.assert_called_once()` and check `mock.call_args.kwargs['max_results'] == 3`

Target: 70/70 research tests. Update `testing/TESTING_TODO.md` baselines. Commit + push.

---

## Task 2 — Test consolidation (Alchemy)

See `TODO.md` Task B for full spec. The key items a laptop can do:

1. Create `tests/test_apu/conftest.py` with shared `get_real_vram()`, `assert_vram_safe()`, `vram_snapshot_str()` — pick the best implementation from the 4 copies
2. Update imports in `test_vram_live.py`, `test_apu_concurrency_live.py`, `test_apu_priority_live.py`, `test_apu_integration.py` to use conftest
3. Remove voice latency tests from `test_apu_concurrency_live.py` (priority_live owns them)
4. Deepen `test_apu_api.py` with error cases and response validation

Commit + push.

---

## Task 3 — BaratzaMemory test helpers to conftest

In BaratzaMemory repo:

1. Move `_make_memory_row()` from test_search.py and test_behavioral.py → `tests/conftest.py`
2. Move `_make_qdrant_result()` → `tests/conftest.py`
3. Create `all_mocks` fixture in conftest.py (combines mock_db + mock_qdrant + mock_vault + mock_embeddings)

Commit + push.

---

## Task 4 — Update TESTING_TODO baselines

After Tasks 1-3, update `testing/TESTING_TODO.md`:
- Baselines table: update Alchemy test counts
- Mark consolidation items as done

Commit + push.

---

## Deferred (don't do until above are done)

- **Serpentine flesh-out** — Phase 4 polish, not urgent
- **Synthetic voice scaffold** — needs understanding of TTS/STT APIs, lower priority than fixing tests
- **BaratzaMemory clusters (Task 16-17)** — new feature, do after consolidation

---

## Rules

- No GPU, no Docker required for any of these tasks
- Don't run integration tests — only unit tests and research tests
- Commit after every task
- If TTS/STT APIs are unclear, read the source first, don't guess
