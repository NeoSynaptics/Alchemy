# Alchemy — Claude Session Guide

## What This Is
Core engine — CPU-side heavy lifting. Runs UI-TARS-72B for GUI interaction on a hidden shadow desktop (WSL2 + Xvfb). NEO-TX (the user-facing layer) delegates heavy work here via API.

## Key Paths
- Config: `config/settings.py`
- Server: `alchemy/server.py` (FastAPI, port 8000)
- Shadow Desktop: `alchemy/shadow/` (WSL2 bridge, controller, health)
- Vision Agent: `alchemy/agent/` (screenshot → UI-TARS → action → xdotool loop)
- Models: `alchemy/models/` (CPU model lifecycle)
- Router: `alchemy/router/` (request classification)
- Auth: `alchemy/security/` (bearer tokens)
- WSL Scripts: `wsl/` (setup, start, stop, health)

## Model
- **UI-TARS-72B** → CPU (128GB RAM) → GUI visuomotor agent (~42GB, 3-5 tok/s)

## What Alchemy Does NOT Own
- Voice (STT/TTS) → NEO-TX (GPU)
- User conversation → NEO-TX (14B conversational model on GPU)
- Tray widget → NEO-TX
- Approval gates → NEO-TX
- GPU models → NEO-TX

## API
- `POST /vision/analyze` → screenshot → UI-TARS → action JSON
- `POST /vision/task` → submit multi-step GUI task
- `POST /shadow/start|stop` → control shadow desktop
- `GET /shadow/health` → service status
- `GET /models` → model status + RAM usage

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
NEO-TX (port 8100) sends GUI tasks to Alchemy. For APPROVE-tier actions, the agent loop pauses and sends an approval request back to NEO-TX before executing.

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
