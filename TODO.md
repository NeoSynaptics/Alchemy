# Alchemy — Current Tasks

Read this file when you start. Do tasks in order, top to bottom. Skip tasks marked [DONE]. Commit and push after each task.

**Repo:** `C:\Users\monic\Documents\Alchemy_explore` (branch: main)

---

## [DONE] Task 1: Register NEOSY models in gpu_fleet.yaml
## [DONE] Task 2: Add NEOSY settings to Alchemy config
## [DONE] Task 3: Mount NEOSY as sub-app in server.py
## [DONE] Task 4: APU stabilization and self-healing

---

## Suggested Next Tasks (from codebase review 2026-03-10)

Codebase health: **8.5/10** — 25,973 lines, 90 test files, 18 modules with manifests, 1020+ tests.
No TODO/FIXME/HACK comments. API error handling is solid. Main gap = missing tests for 3 automation APIs.

### Task 5 [HIGH]: Add tests for `/v1/click/*` API endpoints
- Create `tests/test_click_api.py`
- Cover: `/v1/click/call`, `/v1/click/flow`, `/v1/click/browser`, `/v1/click/functions`
- Mock Ollama + Playwright, test contract guard, error paths, streaming
- Reference existing test patterns in `tests/test_apu/`

### Task 6 [HIGH]: Add tests for `/v1/desktop/*` API endpoints
- Create `tests/test_desktop_api.py`
- Cover: `/v1/desktop/task`, `/v1/desktop/summon`, `/v1/desktop/dismiss`, `/v1/desktop/mode`
- Mock SendInput + screenshot capture, test mode switching

### Task 7 [MEDIUM]: Add tests for `/gate/review` API endpoint
- Create `tests/test_gate_api.py`
- Cover: gate review with approve/deny/timeout, fail-open behavior

### Task 8 [LOW]: Wire APU health_check into /health and add periodic reconciliation
- Expose `GET /v1/apu/health` calling `orchestrator.health_check()`
- Add optional periodic VRAM reconciliation (e.g. every 60s via background task)
- This builds on the health_check() + reconcile_vram() added in Task 4

### Task 9 [LOW]: Clean up legacy `alchemy/playwright/` module
- Investigate if anything imports from it (vs `alchemy/core/` which has the active Playwright agent)
- If unused, remove or archive
