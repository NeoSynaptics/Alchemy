# APU Orchestrator Changelog

## 2026-03-10 — Stabilization & Self-Healing

### Bugs Found & Fixed

1. **Race conditions on model state transitions**
   - `load_model`, `unload_model`, `demote`, `ensure_loaded` had no synchronization.
   - Concurrent calls could evict the same model twice or corrupt VRAM accounting.
   - **Fix:** Added `asyncio.Lock` (`_state_lock`) guarding all state-mutating operations.

2. **No rollback on failed model loads**
   - `_make_room` evicts models to free VRAM, but if `_backend_load` then fails, evicted models stay demoted — VRAM is wasted and models are unnecessarily offline.
   - **Fix:** On backend load failure, attempt to re-load evicted models back to their GPU slot.

3. **VRAM tracking drift**
   - Registry tracked model locations internally but never cross-checked against actual nvidia-smi / Ollama state. Over time, registry could drift from reality (e.g., Ollama auto-evicting models after keep_alive timeout).
   - **Fix:** Added `reconcile_vram()` that queries Ollama `/api/ps` and corrects registry state.

4. **No startup reconciliation with Ollama**
   - On boot, the orchestrator assumed a clean slate and loaded frozen baseline models without checking what Ollama already had loaded. This could double-load or fail to track pre-existing models.
   - **Fix:** `reconcile_on_startup()` runs before frozen baseline restore, syncing registry with Ollama's actual state.

5. **No health check method**
   - No way to programmatically assess APU health.
   - **Fix:** Added `health_check()` returning healthy/unhealthy status, VRAM drift detection, and Ollama sync state.

### New Methods
- `health_check()` — Full health assessment (VRAM drift, Ollama sync, model counts)
- `reconcile_vram()` — Fix drift between registry and Ollama actual state
- `reconcile_on_startup()` — Auto-runs on boot before baseline restore
- `_ollama_list_running()` — Query Ollama /api/ps for loaded models
