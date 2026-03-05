# Alchemy — Claude Session Guide

## What This Is
Core engine for the NEO stack. GPU/CPU orchestrator, click agent, voice pipeline, research browser, and more. NEO-TX (the user-facing layer) delegates heavy work here via API.

## Key Paths
- Config: `config/settings.py` (nested BaseModel groups per module)
- Server: `alchemy/server.py` (FastAPI, port 8000)
- AlchemyClick: `alchemy/click/` (two-tier GUI automation: Playwright + vision fallback)
- Shadow Desktop: `alchemy/shadow/` (WSL2 bridge, controller, health)
- GPU Orchestrator: `alchemy/gpu/` (VRAM/RAM fleet management, model placement)
- Models: `alchemy/models/` (CPU model lifecycle)
- Router: `alchemy/router/` (request classification + context routing)
- Core: `alchemy/core/` (Playwright agent, browser manager, approval gate)
- Auth: `alchemy/security/` (bearer tokens)
- WSL Scripts: `wsl/` (setup, start, stop, health)

## Module Registry (13 modules)

| Module | ID | Tier | Path |
|--------|----|------|------|
| Agent Kernel | `core` | core | `alchemy/core/` |
| AlchemyClick | `click` | core | `alchemy/click/` |
| Voice Pipeline | `voice` | core | `alchemy/voice/` |
| GPU Orchestrator | `gpu` | infra | `alchemy/gpu/` |
| LLM Adapters | `adapters` | infra | `alchemy/adapters/` |
| Shadow Desktop | `shadow` | infra | `alchemy/shadow/` |
| Context Router | `router` | infra | `alchemy/router/` |
| Cloud AI Bridge | `cloud` | infra | `alchemy/cloud/` |
| Desktop Agent | `desktop` | app | `alchemy/desktop/` |
| Gate Reviewer | `gate` | app | `alchemy/gate/` |
| AlchemyBrowser | `research` | app | `alchemy/research/` |
| AlchemyWord | `word` | app | `alchemy/word/` |

## GPU Fleet & Eviction

- **No model is immune.** All models can be evicted from VRAM.
- **Eviction order:** app models first → infra → core last.
- **Evicted models go to RAM (warm)**, not disk. Fast reload when needed.
- `module_tier` field on ModelCard controls eviction ordering.
- `ModelTier` enum: RESIDENT(P0) > USER_ACTIVE(P1) > AGENT(P2) > WARM(P3) > COLD(P4)

## API
- `POST /vision/analyze` → screenshot → VLM → action JSON
- `POST /vision/task` → submit multi-step GUI task
- `POST /shadow/start|stop` → control shadow desktop
- `GET /shadow/health` → service status
- `GET /v1/modules` → module discovery + contract status
- `POST /v1/gpu/app/{name}/activate-manifest` → resolve model contracts
- `GET /v1/gpu/status` → GPU/RAM status

## Commands
```bash
make shadow-setup    # Install Xvfb/Fluxbox/x11vnc/noVNC in WSL2
make shadow-start    # Start shadow desktop
make shadow-stop     # Stop shadow desktop
make shadow-health   # Check services
make server          # Run Alchemy on :8000
make test            # Run pytest
```

## NEO-TX Integration
NEO-TX (port 8100) sends GUI tasks to Alchemy. For APPROVE-tier actions, the click agent loop pauses and sends an approval request back to NEO-TX before executing.

## Module Conventions (MANDATORY)

### Adding a New Module
1. Create `alchemy/<name>/` with `__init__.py` (public API + `__all__`)
2. Add `alchemy/<name>/manifest.py` with `MANIFEST = ModuleManifest(...)`
3. Add settings group to `config/settings.py` as a nested `BaseModel`
4. Add `<name>_enabled: bool` toggle if the module is optional
5. Add import boundary contract in `.importlinter`
6. Add tests mirroring module structure in `tests/`

### Settings Rules
- One file: `config/settings.py`. Nested `BaseModel` groups per module.
- New code uses nested form: `settings.gate.enabled` (not `settings.gate_enabled`)
- Flat fields kept for backward compat only — do not add new flat fields.
- Every optional module has an `<id>_enabled: bool` toggle.
- Secrets/API keys go in manifest `env_keys`, not hardcoded.

### Import Rules
- Core and adapters NEVER import from features or API
- Feature modules NEVER import from each other (lateral isolation)
- Only `server.py` and `api/` wire features together
- API routes import from their OWN feature module only — no lateral API imports
- GPU/models are infrastructure — no feature imports
- Run `lint-imports` before merging

### Module Registry
- `alchemy/manifest.py` — `ModuleManifest` + `ModelRequirement` frozen dataclasses
- `alchemy/registry.py` — `discover()`, `get()`, `all_modules()`
- `alchemy/contracts.py` — `validate_contracts()`, `ContractReport`
- `GET /v1/modules` — discovery API for setup wizards and settings pages
- Every manifest MUST have: `id`, `name`, `description`, `tier`

### Model Contracts (App → Core)
Apps declare what models they need. Core validates and decides placement.
- Add `models=[ModelRequirement(...)]` to the manifest
- `capability` — what the app needs: "vision", "reasoning", "coding", "embedding", etc.
- `required=True` — app cannot function without this. `False` = nice-to-have.
- `preferred_model` — hint (e.g. "qwen2.5vl:7b"). Core may substitute.
- `min_tier` — minimum tier: "resident", "warm", "cold". Core checks availability.
- `context_tokens` — hint for VRAM budgeting (e.g. 2048 for vision models).
- `alchemy/contracts.py` validates all contracts against `ModelRegistry` on startup.
- Apps NEVER load models directly. They declare needs, core provides.
