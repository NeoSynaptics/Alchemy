# Alchemy

**Core engine — GPU/CPU orchestrator, AlchemyClick (GUI automation), voice pipeline, and more.**

Alchemy is the core backend for the NEO stack. It manages GPU/CPU model fleets, runs AlchemyClick for GUI automation (Playwright + vision fallback), and hosts modular AI apps. NEO-TX (the user-facing layer) delegates heavy work here via API.

## Design Principles

- **Local-first.** Everything runs on your hardware. No cloud dependency.
- **CPU-powered.** 72B model on 128GB RAM. Slow but accurate. Accuracy beats speed for GUI work.
- **Headless.** No user interface. The shadow desktop is invisible. NEO-TX handles all user interaction.

## What Alchemy Owns

| Responsibility | Detail |
|----------------|--------|
| **Shadow Desktop** | WSL2 + Xvfb + Fluxbox + x11vnc + noVNC — hidden virtual desktop |
| **AlchemyClick** | Two-tier GUI automation: Playwright a11y tree + Qwen2.5-VL vision fallback |
| **Click Loop** | screenshot → VLM → parse action → xdotool → repeat |
| **Model Management** | CPU model lifecycle (load/unload/health) |
| **API** | FastAPI on port 8000 — NEO-TX connects here |
| **Auth** | Bearer tokens |

## What Alchemy Does NOT Own

| Responsibility | Owner |
|----------------|-------|
| Voice (STT/TTS) | **NEO-TX** (GPU, fast) |
| User conversation | **NEO-TX** (14B conversational model) |
| Tray widget / viewport | **NEO-TX** |
| Approval gates | **NEO-TX** |
| GPU models | **NEO-TX** |

## Model

| Model | Size | Hardware | Speed | Purpose |
|-------|------|----------|-------|---------|
| **UI-TARS-72B** (Q4_K_M) | ~42GB | CPU (128GB RAM) | 3-5 tok/s | GUI agent — screenshot in, click/type out |

PW/VLM (ByteDance, open-weight) is purpose-built for computer use. 72B is slow on CPU but accurate — each output is a short action JSON, not a novel. The 128GB RAM is the moat.

## Architecture

```
┌───────────────────────────────────────────────┐
│                 Alchemy Core                  │
│                 port 8000                     │
│                                               │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │  Shadow  │  │  Alchemy │  │   Model    │  │
│  │  Desktop │  │  Click   │  │   Manager  │  │
│  │          │  │          │  │            │  │
│  │  WSL2    │  │  PW/VLM │  │  load      │  │
│  │  Xvfb    │  │  loop    │  │  unload    │  │
│  │  xdotool │  │  actions │  │  health    │  │
│  └────┬─────┘  └────┬─────┘  └─────┬──────┘  │
│       │             │              │          │
│  ┌────▼─────────────▼──────────────▼──────┐   │
│  │      Ollama CPU (localhost:11434)      │   │
│  │      UI-TARS-72B Q4_K_M (~42GB RAM)   │   │
│  └────────────────────────────────────────┘   │
└──────────────────┬────────────────────────────┘
                   │ HTTP API
                   ▼
               NEO-TX (:8100)
              (smart interface)
```

## API Endpoints

```
# Health
GET  /health                    → {"status": "ok", "model": "ui-tars:72b"}

# AlchemyClick
POST /vision/analyze            → Send screenshot, get action JSON
POST /vision/task               → Submit a full GUI task (multi-step click loop)
GET  /vision/task/{id}/status   → Check task progress

# Shadow Desktop
POST /shadow/start              → Start shadow desktop
POST /shadow/stop               → Stop shadow desktop
GET  /shadow/health             → Shadow desktop service status
GET  /shadow/screenshot         → Capture current screenshot

# Model Management
GET  /models                    → Model status + RAM usage
POST /models/load               → Load model
POST /models/unload             → Unload model
```

## Quick Start

```bash
# 1. Clone
git clone https://github.com/NeoSynaptics/Alchemy.git
cd Alchemy

# 2. Install
pip install -e .

# 3. Setup WSL2 shadow desktop
make shadow-setup

# 4. Pull model
ollama pull ui-tars:72b  # or manual GGUF import

# 5. Run
make server              # → http://localhost:8000
make shadow-start        # → shadow desktop at localhost:6080
```

## Hardware Requirements

- **RAM:** 128GB (UI-TARS-72B Q4_K_M = ~42GB)
- **CPU:** i9-13900K or equivalent (CPU inference)
- **WSL2:** Ubuntu with Xvfb, Fluxbox, x11vnc, xdotool, scrot

## Connected Projects

- **[NEO-TX](https://github.com/NeoSynaptics/NEO-TX)** — Smart AI interface. Voice, tray widget, approval gates, fast GPU models. Sends GUI tasks to Alchemy.

## License

MIT — NeoSynaptics
