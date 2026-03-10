# Alchemy — Claude Session Guide

## What This Is
Core AI engine. Voice pipeline, click automation, GPU orchestration, research browser, and more. Everything runs from a single server on port 8000.

**Repo:** `C:\Users\monic\Documents\Alchemy_explore` (branch: main)
**GitHub:** NeoSynaptics/Alchemy

## Key Paths
- Config: `config/settings.py` (nested BaseModel groups per module)
- Server: `alchemy/server.py` (FastAPI, port 8000)
- **AlchemyVoice: `alchemy/voice/` (voice pipeline, smart router, conversation, tray, constitution)**
- AlchemyClick: `alchemy/click/` (two-tier GUI automation: Playwright + vision fallback)
- Desktop Agent: `alchemy/desktop/` (native Windows automation: SendInput, ghost cursor)
- APU (Alchemy Processing Unit): `alchemy/apu/` (VRAM/RAM fleet management, model placement, health guard)
- Models: `alchemy/models/` (CPU model lifecycle)
- Router: `alchemy/router/` (request classification + context routing)
- Core: `alchemy/core/` (Playwright agent, browser manager, approval gate)
- Auth: `alchemy/security/` (bearer tokens)

## Module Registry

| Module | ID | Tier | Path |
|--------|----|------|------|
| Agent Kernel | `core` | core | `alchemy/core/` |
| AlchemyClick | `click` | core | `alchemy/click/` |
| **AlchemyVoice** | `voice` | core | `alchemy/voice/` |
| APU (Alchemy Processing Unit) | `apu` | infra | `alchemy/apu/` |
| LLM Adapters | `adapters` | infra | `alchemy/adapters/` |
| Context Router | `router` | infra | `alchemy/router/` |
| Cloud AI Bridge | `cloud` | infra | `alchemy/cloud/` |
| Desktop Agent | `desktop` | app | `alchemy/desktop/` |
| Gate Reviewer | `gate` | app | `alchemy/gate/` |
| AlchemyBrowser | `research` | app | `alchemy/research/` |
| AlchemyWord | `word` | app | `alchemy/word/` |

## AlchemyVoice (alchemy/voice/)
Voice is a first-class core module. Public interface hides model internals.

**Public API** (what external code uses):
- `VoiceSystem` — start/stop/configure voice
- `VoiceMode` — conversation, command, dictation, muted
- `VoiceStatus` — serializable status for GUI/API

**Internal modules** (hidden behind VoiceSystem):
- `pipeline.py` — state machine (IDLE->LISTENING->RECORDING->PROCESSING->SPEAKING)
- `stt.py` — Whisper STT
- `tts.py` — Piper/Fish Speech/Kokoro TTS
- `wake_word.py` — openWakeWord detection
- `vram_manager.py` — single/dual GPU VRAM swapping
- `models/` — voice model registry + providers (Ollama, Alchemy)
- `router/` — smart routing (conversation vs GUI task)
- `constitution/` — approval gates (AUTO/NOTIFY/APPROVE)
- `planner/` — task decomposition
- `tray/` — system tray UI (PyQt6, optional)
- `knowledge/` — NEO-RX integration (optional)
- `api/` — /chat, /voice, /callbacks endpoints

## APU Fleet & Eviction

- **No model is immune.** All models can be evicted from VRAM.
- **Eviction order:** app models first -> infra -> core last.
- **Evicted models go to RAM (warm)**, not disk. Fast reload when needed.
- `module_tier` field on ModelCard controls eviction ordering.
- `ModelTier` enum: RESIDENT(P0) > USER_ACTIVE(P1) > AGENT(P2) > WARM(P3) > COLD(P4)

## API
- `POST /v1/vision/analyze` — screenshot -> VLM -> action JSON
- `POST /v1/vision/task` — submit multi-step GUI task
- `GET /v1/modules` — module discovery + contract status
- `POST /v1/apu/app/{name}/activate-manifest` — resolve model contracts
- `GET /v1/apu/status` — GPU/RAM status
- `POST /v1/chat` — non-streaming chat
- `POST /v1/chat/stream` — SSE streaming chat
- `GET /v1/voice/status` — voice pipeline status
- `POST /v1/voice/start|stop` — control voice pipeline
- `POST /v1/voice/mode` — change voice mode
- `POST /v1/callbacks/approval|notify|task-update` — internal callbacks

## Commands
```bash
make server          # Run Alchemy on :8000
make test            # Run pytest
```

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

### Model Contracts (App -> Core)
Apps declare what models they need. Core validates and decides placement.
- Add `models=[ModelRequirement(...)]` to the manifest
- `capability` — what the app needs: "vision", "reasoning", "coding", "embedding", etc.
- `required=True` — app cannot function without this. `False` = nice-to-have.
- `preferred_model` — hint (e.g. "qwen2.5vl:7b"). Core may substitute.
- `min_tier` — minimum tier: "resident", "warm", "cold". Core checks availability.
- `context_tokens` — hint for VRAM budgeting (e.g. 2048 for vision models).
- `alchemy/contracts.py` validates all contracts against `ModelRegistry` on startup.
- Apps NEVER load models directly. They declare needs, core provides.
