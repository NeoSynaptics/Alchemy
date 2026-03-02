# Alchemy — Claude Session Guide

## What This Is
Local-first LLM core engine. Manages Ollama models, routes requests, handles voice I/O, exposes API on port 8000. Serves NEO-TX and future tools.

## Key Paths
- Config: `config/settings.py`
- Server: `alchemy/server.py` (FastAPI, port 8000)
- Router: `alchemy/router/` (classifier, gateway, escalation)
- Models: `alchemy/models/` (Ollama manager, VRAM scheduling)
- Voice: `alchemy/voice/` (wake word, STT, TTS, pipeline)
- Auth: `alchemy/security/` (bearer token, trust levels)

## Models
- **UI-TARS-72B** → CPU (128GB RAM) → GUI agent (NEO-TX Model B)
- **Qwen2.5-Coder-14B** → GPU (12GB VRAM) → planner, reasoning, voice interpretation
- **Qwen3-8B** → GPU (swapped) → fast chat, triviality

## Voice Pipeline
Voice lives here (NOT in NEO-TX). Flow:
1. Mic → openWakeWord ("Hey Neo")
2. faster-whisper STT (GPU, on-demand ~3GB VRAM)
3. 14B interprets intent
4. Routes: GUI tasks → NEO-TX, text answers → Piper TTS → speaker

## API
- `POST /chat` → route to best model
- `POST /vision/analyze` → screenshot → UI-TARS → action JSON
- `POST /voice/transcribe` → audio → text
- `POST /voice/speak` → text → audio
- `GET /models` → loaded models + resource usage
- `POST /models/load` / `POST /models/unload` → manage models

## Commands
```bash
pip install -e .            # Install core
pip install -e ".[voice]"   # With voice deps
python -m alchemy           # Run server on :8000
pytest tests/ -v            # Test
```

## Connected Projects
- NEO-TX (Shadow Desktop) connects via HTTP to port 8000
- NEO-TX never touches audio — receives pre-parsed intent via API
