# Alchemy — Current Tasks

Read this file when you start. Do tasks in order, top to bottom. Skip tasks marked [DONE]. Commit and push after each task.

**Repo:** `C:\Users\monic\Documents\Alchemy_explore` (branch: main)

---

## [DONE] Task 1: Register NEOSY models in gpu_fleet.yaml
## [DONE] Task 2: Add NEOSY settings to Alchemy config
## [DONE] Task 3: Mount NEOSY as sub-app in server.py
## [DONE] Task 4: APU stabilization and self-healing

---

## [DONE] Task 5: APU Debug Logger & Event Stream

The APU needs structured runtime diagnostics. Not just text logs — a queryable event history so we can debug issues after the fact.

**Create `alchemy/apu/event_log.py`:**

```python
@dataclass
class APUEvent:
    timestamp: datetime
    event_type: str          # "load", "unload", "evict", "promote", "demote", "drift", "error", "reconcile"
    model_name: str | None
    gpu_index: int | None
    app_name: str | None     # who requested this operation
    vram_before_mb: int      # VRAM used on target GPU before operation
    vram_after_mb: int       # VRAM used on target GPU after operation
    vram_expected_mb: int    # what we expected based on model card
    duration_ms: float       # how long the operation took
    success: bool
    error: str | None
    details: dict            # flexible — eviction reasons, candidates considered, etc.
```

**What to log (instrument these in orchestrator.py):**
- Every `load_model` / `unload_model` / `demote` / `evict_to_disk` call with timing
- Every eviction decision: which candidates were considered, why the winner was picked, how much VRAM was freed
- Every `ensure_loaded` call: was it a cache hit (already on GPU) or a load?
- Every `reconcile_vram` run: what drift was detected, what was corrected
- Every `app_activate` / `app_deactivate`: which models, which app, timing
- Every error with full context (what was attempted, what failed, what state was left in)
- VRAM drift events: tracked vs actual (from health_check)

**Ring buffer:** Keep last 500 events in memory (deque). Queryable via:
- `GET /v1/apu/events` — returns recent events, filterable by type/model/app
- `GET /v1/apu/events/errors` — only errors and drift events

**Also add to each event:**
- `expected_duration_ms` — based on model size (rough: 100ms per GB for Ollama loads)
- Flag if `actual > 2x expected` (slowness indicator)

**Files to modify:**
- Create `alchemy/apu/event_log.py` (new)
- Edit `alchemy/apu/orchestrator.py` — instrument all state-changing methods
- Edit APU API routes — add `/v1/apu/events` endpoint

**Commit when done. This is the foundation — Tasks 6 and 7 depend on it.**

---

## [DONE] Task 6: APU Concurrency Fix — Atomic Decide-Evict-Load

The current `_state_lock` doesn't cover the full critical path. The decide→evict→load sequence must be atomic.

**Problem:**
Two concurrent `ensure_loaded()` calls can both check VRAM, both decide to evict the same model, both evict it. Result: VRAM accounting corruption.

**Fix in `alchemy/apu/orchestrator.py`:**

1. **Make `ensure_loaded()` hold `_state_lock` for the ENTIRE operation** — from VRAM check through eviction through load. Not just per-attempt.

2. **Make `_make_room()` run inside the same lock hold** — currently it's called from `_load_model_unlocked()` which is already inside the lock, but verify the lock is held continuously (no await that releases it between evict and load decisions).

3. **Fix `_backend_unload` failure handling** — if unload fails, DON'T mark the model as CPU_RAM. Leave it as GPU. Log an error event.

4. **Fix rollback on load failure** — current rollback tries to re-load evicted models but uses `_backend_load` directly. Use `_load_model_unlocked` with proper VRAM accounting.

5. **Add a `_pending_operations` set** — track which models are currently being loaded/unloaded. If a model is in this set, other operations on it must wait.

**Verify with the event log from Task 5** — every concurrent operation should show up as sequential events with no overlapping timestamps on the same GPU.

**Files to modify:**
- `alchemy/apu/orchestrator.py` — rework locking strategy
- `alchemy/apu/registry.py` — add thread-safety (wrap `_models` dict mutations in a lock)

---

## [DONE] Task 7: APU Test Harness — Concurrency, Failures, and Invariants

Create a comprehensive test harness that catches bugs at development time AND runtime.

**Create `tests/test_apu/test_apu_stress.py`:**

### Concurrency tests (at least 10):
1. 10 concurrent `ensure_loaded()` calls for different models on same GPU — verify no double-eviction
2. 5 concurrent `load_model()` + 5 concurrent `unload_model()` — verify VRAM accounting stays consistent
3. 2 concurrent `app_activate()` calls requesting same GPU — verify no corruption
4. `app_activate()` during `restore_frozen_baseline()` — verify no conflict
5. `reconcile_vram()` during active `load_model()` — verify no state corruption
6. Rapid `promote()` + `demote()` cycles (100 iterations) — verify model ends up in correct state
7. `ensure_loaded()` on model that Ollama silently evicted — verify re-load happens
8. Two `profile_model()` calls simultaneously — verify lock prevents concurrent profiling
9. `app_deactivate()` during `ensure_loaded()` of app's model — verify clean handling
10. `health_check()` during active eviction — verify no crash

### Failure injection tests:
1. Ollama load fails mid-operation — verify evicted models are restored (rollback)
2. Ollama unload fails — verify model stays marked as GPU (not falsely marked RAM)
3. Ollama goes completely unreachable — verify graceful degradation
4. GPU reports 0 free VRAM (nvidia-smi) — verify no infinite eviction loop
5. Frozen baseline references deleted model — verify clean skip + warning

### Invariant checks (run after EVERY test):
```python
async def assert_apu_invariants(orch):
    """Run after every test to verify APU state is consistent."""
    # 1. VRAM accounting: sum of model cards on GPU <= GPU total
    for gpu_idx in [0, 1]:
        total_on_gpu = orch._registry.total_vram_on_gpu(gpu_idx)
        gpu_info = await orch._monitor.get_gpu(gpu_idx)
        assert total_on_gpu <= gpu_info.total_vram_mb, f"GPU {gpu_idx} VRAM overcommitted"

    # 2. No model in two locations at once
    all_models = orch._registry.all_models()
    for model in all_models:
        locations = []
        if model.current_location == ModelLocation.GPU_0: locations.append("GPU_0")
        if model.current_location == ModelLocation.GPU_1: locations.append("GPU_1")
        if model.current_location == ModelLocation.CPU_RAM: locations.append("RAM")
        if model.current_location == ModelLocation.DISK: locations.append("DISK")
        assert len(locations) == 1, f"{model.name} in {len(locations)} locations"

    # 3. No negative VRAM values
    for model in all_models:
        assert model.vram_mb >= 0, f"{model.name} has negative VRAM"

    # 4. Tier consistency: RESIDENT models should be on GPU (warn if not)
    for model in all_models:
        if model.default_tier == ModelTier.RESIDENT and not model.current_location.is_gpu:
            warnings.warn(f"Resident model {model.name} not on GPU")

    # 5. Event log has no gaps (if event_log exists)
    if hasattr(orch, '_event_log'):
        events = orch._event_log.recent()
        for e in events:
            assert e.timestamp is not None
            assert e.event_type in VALID_EVENT_TYPES
```

### Runtime invariant checker:
Also create `alchemy/apu/invariants.py` — a module that can be called periodically (every 60s) or after every APU operation to verify state consistency. Log violations as APU events with type="invariant_violation". This catches bugs in production, not just in tests.

**Create `tests/test_apu/test_apu_diagnostics.py`:**
- Test that event_log captures all operation types
- Test that events have correct VRAM before/after values
- Test that slow operations are flagged
- Test that error events contain useful context
- Test that ring buffer doesn't exceed 500 events

**Files to create:**
- `tests/test_apu/test_apu_stress.py` (new)
- `tests/test_apu/test_apu_diagnostics.py` (new)
- `alchemy/apu/invariants.py` (new)

**Files to modify:**
- `alchemy/apu/orchestrator.py` — call invariant checker after state changes (behind a debug flag)

---

## [DONE] Task 8: Wire APU health_check + periodic reconciliation into server

After Tasks 5-7 are solid:
- Expose `GET /v1/apu/health` calling `orchestrator.health_check()`
- Add background task in server.py lifespan: run `reconcile_vram()` every 60 seconds
- Add background task: run `invariants.check()` every 60 seconds (log violations, don't crash)
- Add `GET /v1/apu/events` endpoint from Task 5

**Files to modify:**
- `alchemy/server.py` — add background tasks in lifespan
- APU API routes — add health and events endpoints

---

## Task 9: APU Concurrency Fixes — Review Findings

Code review found bugs in the Tasks 5-8 implementation. Fix these before moving on.

**Bug 1: `_backend_unload` failure does NOT prevent state change (CRITICAL)**

In `demote()` and `unload_model()`, `_backend_unload(card)` is called but the return value is ignored. The model gets marked as CPU_RAM/DISK even if the backend unload failed — leaving state corrupted (registry says RAM, GPU still has it loaded).

Fix in `alchemy/apu/orchestrator.py`:
```python
# demote() — BEFORE (broken):
if card.current_location.is_gpu:
    await self._backend_unload(card)  # return value ignored!
self._registry.update_location(name, ModelLocation.CPU_RAM, ModelTier.WARM)

# demote() — AFTER (fixed):
if card.current_location.is_gpu:
    ok = await self._backend_unload(card)
    if not ok:
        self._event_log.record("error", model_name=name, success=False,
                               error="Backend unload failed, keeping GPU state")
        return False
self._registry.update_location(name, ModelLocation.CPU_RAM, ModelTier.WARM)
```

Apply same pattern to `unload_model()`.

**Bug 2: `_pending_operations` only guards loads, not unloads**

`_pending_operations` is checked in `_load_model_unlocked` but NOT in `unload_model()` or `demote()`. A concurrent load and unload on the same model can race.

Fix: Add `_pending_operations` check to `unload_model()` and `demote()` — if model is in the set, return False.

**Bug 3: Missing test — concurrent `ensure_loaded` on SAME model**

The stress test tests 10 different models. The most dangerous race is two `ensure_loaded()` on the SAME model. Add a test that verifies the second call gets the "busy" error from the `_pending_operations` guard.

**Bug 4: `test_unload_failure_keeps_gpu_state` doesn't assert GPU state**

The test name says "keeps GPU state" but it only checks event logging. Add assertion:
```python
assert card.current_location.is_gpu, "Model should stay on GPU after unload failure"
```

**Bug 5: Missing test from spec — `app_activate` during `restore_frozen_baseline`**

Concurrency test #4 from the Task 7 spec was skipped. Add it.

**Files to modify:**
- `alchemy/apu/orchestrator.py` — fix `demote()` and `unload_model()` to check `_backend_unload` return value + add `_pending_operations` guards
- `tests/test_apu/test_apu_stress.py` — fix existing test + add 3 missing tests

**Commit when done.**

---

## Completed (from previous codebase review)

### [DONE] Click API tests
- `tests/test_click_api.py` — 11 tests: auto-routing, flow, browser, functions, contract guard

### [DONE] Desktop API tests
- `tests/test_desktop/test_desktop_api.py` — 15 tests: task submit, polling, summon/dismiss, mode, 503s, contract guard

### [DONE] Gate API tests
- `tests/test_gate/test_gate_api.py` — 10 tests: accept/deny/other, fail-open, timeout, validation, contract guard

### [DONE] Legacy playwright cleanup
- Removed `alchemy/playwright/` and `tests/test_playwright/` — no production imports
- Cleaned up `.importlinter` reference
