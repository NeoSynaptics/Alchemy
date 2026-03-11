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

## [DONE] Task 9: APU Concurrency Fixes — Review Findings

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

## [DONE] Task 10: APU Audit Fixes — Rollback Bug, Test Quality, Invariant Gaps

Code review found real issues in the Tasks 5-8 implementation that need fixing before we build on top.

**Fix 1: Rollback restores wrong tier (CRITICAL)**

In `alchemy/apu/orchestrator.py`, when load fails after evicting models, rollback uses `evicted_card.default_tier` — but the evicted model may have been RESIDENT. Restoring as WARM violates invariants.

```python
# BEFORE (broken) — in _load_model_inner rollback block:
self._registry.update_location(evicted_name, loc, evicted_card.default_tier)

# AFTER (fixed) — save original tier BEFORE eviction:
# At eviction time, store: evicted_tiers[name] = card.current_tier
# At rollback time:
self._registry.update_location(evicted_name, loc, evicted_tiers[evicted_name])
```

Also handle partial rollback: if re-loading the 3rd evicted model fails, log which models couldn't be restored and mark them as needing manual reconciliation.

**Fix 2: Tests need real VRAM state tracking**

Current stress tests mock `snapshot()` to always return the same values — loads/unloads never change VRAM readings. The mock needs to track state:

```python
class FakeGPUMonitor:
    def __init__(self):
        self.gpu_used = {0: 0, 1: 0}  # track VRAM per GPU

    async def snapshot(self):
        return GPUSnapshot(gpus=[
            GPUInfo(index=0, total_vram_mb=12000, used_vram_mb=self.gpu_used[0]),
            GPUInfo(index=1, total_vram_mb=16000, used_vram_mb=self.gpu_used[1]),
        ])
```

Wire `_backend_load` mock to increment `gpu_used` by model's `vram_mb`, and `_backend_unload` to decrement it. Then invariant checks will actually catch VRAM overcommit.

**Fix 3: Invariant checker — add missing checks**

In `alchemy/apu/invariants.py`:
- Add: RESIDENT tier model NOT on GPU → violation (currently only checks reverse)
- Remove: dead null-check on `current_location` (it defaults to DISK, never None)
- Add: total tracked VRAM on GPU vs snapshot actual VRAM — flag drift > 500MB

**Fix 4: Lock granularity comment**

The global `_state_lock` blocks ALL model operations while one loads. This is correct but slow. Add a `# TODO: per-GPU lock for better throughput` comment in orchestrator.py. Don't fix now — just document the trade-off.

**Files to modify:**
- `alchemy/apu/orchestrator.py` — fix rollback tier, add TODO comment on lock
- `alchemy/apu/invariants.py` — add RESIDENT check, remove dead code, add VRAM drift check
- `tests/test_apu/test_apu_stress.py` — replace static mock with stateful FakeGPUMonitor
- `tests/test_apu/test_apu_diagnostics.py` — add test for VRAM drift invariant

**Commit when done.**

---

## [DONE] Task 11: Portability Pass — No Hardcoded Paths, Zip-and-Ship Ready

Alchemy must be portable. Right now it's welded to one machine. Fix that.

**Critical hardcoded paths to remove:**

1. `alchemy/server.py` line 498: `sys.path.insert(0, r'C:\Users\monic\BaratzaMemory\src')` — replace with `settings.neosy.src_path` (already have `NeosySettings` in config)
2. `config/settings.py` line 264: `storage_path: str = "D:/AlchemyMemory"` — change default to `./data/memory` (relative)
3. `config/settings.py` line 269: `chroma_path: str = "D:/AlchemyMemory/chroma"` — derive from `storage_path`
4. `scripts/migrate_memory_v2.py` line 20: `DB_PATH = Path("D:/AlchemyMemory/timeline.db")` — use settings

**Settings / env var cleanup:**

5. Add `neosy_src_path: str = ""` to `NeosySettings` — empty = NEOSY not mounted. Set via `ALCHEMY_NEOSY_SRC_PATH` env var.
6. Create `.env.example` at repo root with ALL configurable settings documented:
   - Ports (8000, 11434, 5432, 6333, etc.)
   - Paths (storage, NEOSY src, chroma)
   - GPU settings (device, mode)
   - Credentials (pg_user, pg_password)
   - Model names (all the Qwen/Ollama defaults)
7. Ensure `config/settings.py` reads from env vars for all paths and credentials (most already do via pydantic-settings, verify)

**GPU portability:**

8. `ModelLocation` enum in `alchemy/apu/registry.py` hardcodes `GPU_0` and `GPU_1` — add a comment noting the 2-GPU limit and create a helper `gpu_location(index: int) -> ModelLocation` so the limit is in one place if we expand later. Don't over-engineer — just document the constraint.
9. `config/settings.py` line 363: `click_omniparser_device: str = "cuda:0"` — already configurable, just ensure it's in `.env.example`

**Files to modify:**
- `alchemy/server.py` — remove hardcoded BaratzaMemory path, use settings
- `config/settings.py` — fix default paths to be relative, add neosy_src_path
- `scripts/migrate_memory_v2.py` — use settings instead of hardcoded path
- Create `.env.example` (new)

**Commit when done.**

---

## [DONE] Task 12a: APU Orchestrator — Race Conditions & Missing Locks

Code review found race conditions in orchestrator methods that read/modify state without holding `_state_lock`.

**Bug 1: `rebalance()` reads model state without lock (lines 414-441)**

`rebalance()` iterates `all_models()`, reads `current_location` and `preferred_gpu`, then calls `demote()` and `load_model()`. Between the reads and the lock acquisitions inside those methods, another coroutine can change state.

Fix: Wrap the entire rebalance loop body. Since `demote()` and `load_model()` acquire `_state_lock` internally, either:
- Make internal `_demote_unlocked()` / `_load_model_unlocked()` variants callable from rebalance while it holds the lock, OR
- Snapshot model names + preferred GPUs first, then operate on each sequentially (each call acquires its own lock, state may change but operations are individually safe)

**Bug 2: `user_idle()` mutates collection during iteration (lines 552-559)**

`models_by_tier()` returns a list, but `demote()` changes the model's tier. If `models_by_tier()` ever returns a view instead of a copy, this corrupts iteration.

Fix: Snapshot names before iterating:
```python
model_names = [c.name for c in self._registry.models_by_tier(ModelTier.USER_ACTIVE)]
for name in model_names:
    ok = await self.demote(name)
```

**Bug 3: `app_deactivate()` has no lock protection (lines 523-548)**

`_app_models.pop()` and `card.owner_app = None` happen without lock. Concurrent `app_activate()` for the same app can race with deactivation.

Fix: Wrap in `async with self._state_lock:` or snapshot names first.

**Bug 4: `app_deactivate()` and `user_active()` missing event log (lines 523-572)**

Unlike `app_activate()`, neither records events. No audit trail for app lifecycle.

Fix: Add `event_log.record("app_deactivate", ...)` and `event_log.record("user_active", ...)`.

**Bug 5: Rollback failures not recorded as events (lines 256-278)**

When `_backend_load` fails and rollback of evicted models also fails, only a `logger.warning` is emitted. No event is logged for the rollback failure itself.

Fix: Add `event_log.record("error", ...)` for each rollback failure with context about which model couldn't be restored.

**Files to modify:**
- `alchemy/apu/orchestrator.py` — fix all 5 bugs above

**Commit when done.**

---

## [DONE] Task 12b: Registry Thread-Safety & Event Log Filter Bug

**Bug 1: Registry docstring claims "thread-safe" but has NO locks (registry.py line 105-109)**

`ModelRegistry.__init__` creates a bare `dict` with no lock. The docstring says "Thread-safe model registry" but concurrent calls to `register()`, `unregister()`, `update_location()` can corrupt state. The orchestrator's `_state_lock` only protects orchestrator-level operations — if any code calls registry directly, it's unprotected.

Fix: The registry is protected by the orchestrator's lock in practice, but the docstring is misleading. Either:
- (Minimal) Fix docstring to say "Not inherently thread-safe — callers must hold orchestrator._state_lock"
- (Better) Use `gpu_location()` in `models_on_gpu()` line 136 instead of re-implementing the GPU index mapping inline

**Bug 2: `models_on_gpu()` doesn't use `gpu_location()` helper (registry.py line 135-137)**

```python
loc = ModelLocation.GPU_0 if gpu_index == 0 else ModelLocation.GPU_1  # hardcoded
```

The `gpu_location()` function (line 62) exists as the single point of control for GPU mapping but isn't used here. If GPU_2 is added, `models_on_gpu()` breaks silently.

Fix: Use `gpu_location(gpu_index)` with try/except for invalid index.

**Bug 3: Event log `errors_only` filter includes successful drift events (event_log.py line 140)**

```python
if errors_only and event.success and event.event_type != "drift":
    continue
```

This special-cases drift events to always pass through `errors_only` filter, even when `success=True`. A successful drift event is contradictory — drift detection that succeeded at recovery should NOT appear in error-only views.

Fix: Remove the drift exception. If drift events should always be errors, ensure they're recorded with `success=False`:
```python
if errors_only and event.success:
    continue
```

**Bug 4: Slow detection fails when `vram_expected_mb=0` (event_log.py line 89)**

```python
slow = duration_ms > 2 * expected_ms if expected_ms > 0 else False
```

If `vram_expected_mb=0` (e.g., custom backend models), `expected_ms=0` and `slow` is always `False` — even for 30-second loads. A 5-second load should always be flagged as slow regardless of expected time.

Fix: Add absolute threshold fallback:
```python
_ABSOLUTE_SLOW_MS = 5000.0  # 5 seconds is always slow
slow = (duration_ms > 2 * expected_ms if expected_ms > 0
        else duration_ms > _ABSOLUTE_SLOW_MS)
```

**Files to modify:**
- `alchemy/apu/registry.py` — fix docstring, use `gpu_location()` in `models_on_gpu()`
- `alchemy/apu/event_log.py` — fix `errors_only` filter, add absolute slow threshold

**Commit when done.**

---

## [DONE] Task 12c: APU Test Quality — Weak Assertions & Missing Coverage

Tests pass but several don't actually prove the code works correctly.

**Fix 1: Concurrent load assertion too weak (test_apu_stress.py ~line 135)**

```python
assert len(successes) >= 1  # At least some should succeed
```

10 models × 500MB = 5000MB on GPUs with 28GB total. ALL should succeed. Fix:
```python
assert len(successes) == 10, f"Expected all 10, got {len(successes)}"
```

**Fix 2: Pending operations test doesn't test real concurrency (test_apu_stress.py ~line 237-250)**

Manually inserts into `_pending_operations` set, then calls `_load_model_unlocked()` directly. This tests the guard check but NOT actual concurrent `load_model()` calls racing.

Fix: Use `asyncio.gather()` with two concurrent `load_model()` calls on the same model:
```python
results = await asyncio.gather(
    orch.load_model(card.name, gpu=0),
    orch.load_model(card.name, gpu=0),
)
assert sum(1 for r in results if r.success) == 1
```

**Fix 3: Promote/demote cycles don't verify intermediate state (test_apu_stress.py ~line 176-188)**

After 100 cycles, only checks final state. Add per-cycle assertion that model alternates between GPU and RAM correctly. Also verify VRAM returns to 0 after all demotes (no leak).

**Fix 4: Load failure rollback doesn't verify evicted model was restored (test_apu_stress.py ~line 256-280)**

Test checks `not result.success` but doesn't assert the evicted model is back on GPU. Add:
```python
assert existing.current_location.is_gpu, "Evicted model should be restored after failed load"
```

**Fix 5: Event log assertions too loose (multiple tests)**

`assert len(events) >= 1` will pass with any error event, not necessarily the one being tested. Fix: Filter by `model_name` and check error message content.

**Fix 6: Add missing test — monitor snapshot failure**

No test covers what happens when `monitor.snapshot()` raises an exception (e.g., GPU driver crash). Add a test that makes snapshot raise `RuntimeError` and verify `ensure_loaded()` handles it gracefully.

**Fix 7: Add missing test — negative VRAM invariant**

No test creates a card with negative `vram_mb` to verify the invariant checker catches it.

**Files to modify:**
- `tests/test_apu/test_apu_stress.py` — fixes 1-4, 6
- `tests/test_apu/test_apu_diagnostics.py` — fixes 5, 7

**Commit when done.**

---

## [DONE] Task 12d: Invariant Checker — Missing Checks & Edge Cases

**Fix 1: GPU index bounds not validated (invariants.py line 39-45)**

If pynvml reports GPU indices beyond 0-1, `registry.total_vram_on_gpu()` silently returns 0 (no models match). Add bounds check:
```python
for gpu in snap.gpus:
    if gpu.index > 1:
        violations.append(f"GPU index {gpu.index} found but APU only supports GPU 0-1")
        continue
```

**Fix 2: RESIDENT+RAM not flagged (invariants.py line 60-63)**

Only checks COLD+RAM. But RESIDENT models in RAM is also an invariant violation — RESIDENT means "should always be on GPU."

Fix: Expand the RAM check:
```python
if model.current_location == ModelLocation.CPU_RAM and model.current_tier == ModelTier.RESIDENT:
    violations.append(f"{model.name} in RAM but tier=resident (must be on GPU)")
```

Note: Check #4 (lines 65-70) catches RESIDENT not on GPU, but using `current_tier` not `default_tier`. If a RESIDENT model was temporarily demoted (tier changed to WARM), check #4 won't catch it but check #3 should. Verify both checks complement each other correctly.

**Fix 3: Event log `record()` errors silently swallowed (invariants.py line 89-95)**

If `event_log.record()` raises, the invariant check silently fails to record violations. Wrap in try/except with logger fallback.

**Files to modify:**
- `alchemy/apu/invariants.py` — all 3 fixes

**Commit when done.**

---

## ═══════════════════════════════════════════
## LAPTOP TASKS (no GPU needed) — Tasks 13-23
## ═══════════════════════════════════════════

## Task 13: NEOSY Bug Fixes (port, CLI column, vault path)

Three quick fixes found during code review.

1. `.env.example` says `API_PORT=8000` but `config.py` defaults to 8001 → fix `.env.example`
2. `cli.py` accesses `c["connection_type"]` but DB column is `connection` → fix accessor
3. `vault_path: Path = Path("./vault")` is relative → resolve to absolute in config, or create in lifespan

**Repo:** BaratzaMemory
**Files:** `.env.example`, `src/baratza/cli.py`, `src/baratza/config.py`
**Commit when done.**

---

## Task 14: NEOSY Test Suite — Zero Tests Exist

`tests/` is empty. Add baseline coverage for the core paths.

- `tests/test_ingest.py` — mock DB, test text ingest creates memory + calls embed
- `tests/test_search.py` — mock Qdrant, test search returns ranked results
- `tests/test_pinning.py` — test pin/unpin logic, registro logging
- `tests/test_batch.py` — test batch ingest creates N memories, returns batch_id
- `tests/test_registro.py` — test append-only, verify no mutations

Use `httpx.AsyncClient` + `app` for route tests. Mock DB/Qdrant — don't need real Docker.

**Repo:** BaratzaMemory
**Commit when done.**

---

## [DONE] Task 15: Dashboard Code Wiring (React UI → FastAPI)

Wire the existing React UI to serve from FastAPI in production.

1. Mount `ui/dist/` in `server.py` as catch-all static files (after `/v1/*` routes)
2. Verify `ui/src/pages/Dashboard.tsx` calls correct endpoints (`/v1/apu/status`, `/v1/apu/events`, `/v1/modules`)
3. Add `ui-build` and `dev` targets to Makefile
4. If Dashboard.tsx is a stub, build it out: APU status cards, model table, event feed, module health

**Repo:** Alchemy
**Files:** `alchemy/server.py`, `ui/src/pages/Dashboard.tsx`, `Makefile`
**Do NOT** add Jinja2 or rewrite the React framework.
**Commit when done.**

---

## [DONE] Task 16: Security Module — Currently 1 Line

`alchemy/security/` is just a docstring. Implement basic auth.

1. Bearer token validation middleware (read token from `ALCHEMY_API_TOKEN` env var)
2. Apply to all `/v1/*` routes except `/v1/health`
3. Add `security_enabled: bool = False` toggle in settings (off by default for dev)
4. Add manifest.py for the module
5. Tests: valid token passes, invalid rejected, disabled = no auth

**Repo:** Alchemy
**Files:** `alchemy/security/__init__.py`, `alchemy/security/middleware.py`, `alchemy/security/manifest.py`
**Commit when done.**

---

## [DONE] Task 17: AlchemyWord — Flesh Out the Stub

`alchemy/word/` is 5 lines. Build the basic text generation pipeline.

1. `alchemy/word/writer.py` — text generation via Ollama (qwen3:14b): summarize, rewrite, expand, translate
2. `alchemy/word/api.py` — `POST /v1/word/generate` (prompt + mode → text)
3. Mount in server.py
4. Tests: mock Ollama, verify each mode produces output
5. Keep it simple — no RAG, no memory, just clean text generation with mode selection

**Repo:** Alchemy
**Files:** `alchemy/word/writer.py`, `alchemy/word/api.py`, update `alchemy/word/__init__.py`
**Commit when done.**

---

## [DONE] Task 18: Docker Compose for Alchemy

No docker-compose exists at repo root. Create one for full-stack local dev.

```yaml
services:
  alchemy:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    depends_on: [ollama]
    volumes: ["./data:/app/data"]
  ollama:
    image: ollama/ollama
    ports: ["11434:11434"]
    volumes: ["ollama_data:/root/.ollama"]
    deploy:
      resources:
        reservations:
          devices: [{capabilities: [gpu]}]
```

Also create `Dockerfile` (Python 3.11, pip install, uvicorn CMD).
Test: `docker-compose up` starts both services.

**Repo:** Alchemy
**Files:** `docker-compose.yml` (new), `Dockerfile` (new)
**Commit when done.**

---

## [DONE] Task 19: AlchemyHole Spec — Device Tunnel Design

Write architecture spec only (no code). Like the Instagram saves agent spec.

Save to `alchemy/hole/SPEC.md`:
- Phone/tablet pushes files to PC via HTTP tunnel (local network or Tailscale)
- PC receives into `~/neosy_inbox/hole/` with metadata sidecar
- NEOSY batch ingest picks up files automatically
- Akinator-style query: model asks clarifying questions to find content
- Security: device pairing via one-time code

**Repo:** Alchemy
**Files:** `alchemy/hole/SPEC.md` (new), `alchemy/hole/__init__.py` (empty)
**Commit when done.**

---

## [DONE] Task 20: Voice Test Audit — Do 18 Test Files Cover Real Paths?

Voice has 40 source files and 18 test files. Audit coverage quality.

1. List all test files, check what they test vs what they mock
2. Identify: are STT/TTS/wake word tested with real audio or just mocked?
3. Identify: is the pipeline state machine (IDLE→LISTENING→RECORDING→PROCESSING→SPEAKING) fully tested?
4. Identify: is the smart router tested with real conversation vs GUI task routing?
5. Write a report as `tests/test_voice/COVERAGE_REPORT.md` with gaps and suggested tests

**Repo:** Alchemy
**Read only — don't change voice code. Just document gaps.**
**Commit when done.**

---

## [DONE] Task 21: Import Boundary Audit

Run `lint-imports` and fix any violations. The module conventions say:
- Core/adapters never import from features
- Features never import from each other (lateral isolation)
- Only server.py wires things together

Check if Tasks 5-11 introduced any violations. Fix them.

**Repo:** Alchemy
**Commit when done.**

---

## [DONE] Task 22: README + Setup Guide

No setup guide exists for new users. Create one.

1. `README.md` — what Alchemy is (3 sentences), architecture diagram (ASCII), module list
2. Setup: clone, copy `.env.example`, `pip install`, `make server`
3. Docker: `docker-compose up`
4. Link to each module's purpose (1 line each)
5. Keep it under 100 lines. No marketing fluff.

**Repo:** Alchemy
**Commit when done.**

---

## ═══════════════════════════════════════════
## PC TASKS (needs GPU / Docker / Ollama) — Tasks 23-27
## ═══════════════════════════════════════════

## [DONE] Task 23: Dashboard E2E — Live APU Data

Start the full server, open browser, verify the dashboard shows real GPU data.

1. `make server` → open `localhost:8000`
2. Verify APU status cards show real VRAM numbers from nvidia-smi
3. Verify event feed shows real load/unload events
4. Verify module health indicators reflect actual module state
5. Fix any API mismatches between frontend and backend

**Repo:** Alchemy
**Requires:** GPU, Ollama running

---

## [DONE] Task 24: NEOSY E2E with Ollama — NEO Queue Processing

Task 9 skipped this because Ollama wasn't running. Now test it.

1. Start Docker (PostgreSQL + Qdrant), Ollama with qwen3:14b
2. Ingest 5 test documents via POST /ingest
3. Run POST /neo/queue/process
4. Verify Qwen classified each memory (properties populated)
5. Search for classified content, verify enriched results

**Repo:** BaratzaMemory
**Requires:** Docker, Ollama + qwen3:14b

---

## [DONE] Task 25: Voice Pipeline E2E — Mic to Speaker

Full voice loop test.

1. Start server with voice enabled
2. Test wake word detection (openWakeWord)
3. Test STT: speak → verify transcription
4. Test LLM response: verify Qwen generates coherent reply
5. Test TTS: verify audio output (Piper or Kokoro)
6. Test full loop: wake word → speak → get voice response
7. Test mode switching: conversation vs command vs dictation

**Repo:** Alchemy
**Requires:** GPU (Whisper + TTS models), microphone, speakers

---

## [DONE] Task 26: RLHF Foundation — Reaction Logging for Voice

Wire voice responses + user reactions into NEOSY registro.

1. After voice responds, log: `{action: "voice_response", content: response_text, context: user_query}`
2. Detect user reactions: explicit ("that's funny", "that's dumb") + implicit (laugh detection via audio, silence = boring)
3. Log reactions: `{action: "user_reaction", sentiment: "positive"|"negative"|"neutral", intensity: 0.0-1.0}`
4. Store as NEOSY registro entries linked to the voice response
5. This creates the preference dataset for future DPO/RLHF fine-tuning

**Repo:** Alchemy + BaratzaMemory
**Requires:** GPU, voice pipeline working (Task 25 first)

---

## [DONE] Task 27: Full Integration Smoke Test

Everything up, everything connected, end-to-end.

1. `docker-compose up` (Alchemy + Ollama + PostgreSQL + Qdrant)
2. Open dashboard at `localhost:8000` — verify all green
3. Ingest a document via API → verify it appears in NEOSY
4. Search for it via API → verify ranked results
5. Voice query "what did I just save?" → verify voice response references the document
6. Check APU events show the full model load/unload cycle
7. Check registro has entries for ingest, search, voice interaction

**Repo:** Both
**Requires:** Everything running

---

## ═══════════════════════════════════════════
## Completed
## ═══════════════════════════════════════════

### [DONE] Tasks 1-11
- GPU fleet registration, NEOSY settings, server mount
- APU stabilization, event logger, concurrency fix, stress tests, invariants
- APU health endpoint, periodic reconciliation
- APU concurrency review fixes, audit fixes
- Portability pass (hardcoded paths removed)
- Click/Desktop/Gate API tests, legacy playwright removal

### [DONE] Tasks 12a-12d (Review)
- Orchestrator race conditions: rebalance/user_idle/app_deactivate snapshot-before-iterate
- Event logging added to app_deactivate, user_active, rollback failures
- Registry docstring fixed, gpu_location() used consistently
- Event log: errors_only filter fixed, absolute slow threshold added
- Tests: stronger assertions, concurrent same-model test, snapshot failure test
- Invariant checker: GPU bounds check, RESIDENT+RAM flag, safe event recording

### [DONE] Tasks 15-22 (Laptop Tasks)
- Dashboard: mounted ui/dist/ as catch-all, Makefile targets (ui-build, ui-dev)
- Security: bearer token middleware with toggle, 7 tests
- AlchemyWord: text generation (summarize/rewrite/expand/translate), 10 tests
- Docker: Dockerfile + docker-compose.yml (Alchemy + Ollama with GPU)
- AlchemyHole: device tunnel spec (pairing, upload, Akinator query)
- Voice audit: coverage report (65% unit, 0% real audio, key gaps documented)
- Import boundary: all 11 contracts clean, 0 violations
- README: architecture diagram, setup guide, Docker instructions
