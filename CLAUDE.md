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
