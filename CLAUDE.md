# Alchemy — Claude Session Guide

## What This Is
Local-first LLM core engine. Manages Ollama models, routes requests, exposes API on port 8000. Serves NEO-TX and future tools.

## Key Paths
- Config: `config/settings.py`
- Server: `alchemy/server.py` (FastAPI, port 8000)
- Router: `alchemy/router/` (classifier, gateway, escalation)
- Models: `alchemy/models/` (Ollama manager, VRAM scheduling)
- Auth: `alchemy/security/` (bearer token, trust levels)

## Models
- **UI-TARS-72B** → CPU (128GB RAM) → GUI agent (NEO-TX Model B)
- **Qwen2.5-Coder-14B** → GPU (12GB VRAM) → planner, reasoning
- **Qwen3-8B** → GPU (swapped) → fast chat, triviality

## API
- `POST /chat` → route to best model
- `POST /vision/analyze` → screenshot → UI-TARS → action JSON
- `GET /models` → loaded models + resource usage
- `POST /models/load` / `POST /models/unload` → manage models

## Commands
```bash
pip install -e .         # Install
python -m alchemy        # Run server on :8000
pytest tests/ -v         # Test
```

## Connected Projects
- NEO-TX (Shadow Desktop) connects via HTTP to port 8000
