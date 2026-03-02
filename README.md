# Alchemy

**Local-first LLM core engine. Ollama model management, routing, and shared API.**

Alchemy is the foundation layer that any Alchemy-ecosystem tool connects to. It manages local models via Ollama, routes requests to the right model, and exposes a clean API. Currently serves [NEO-TX](https://github.com/NeoSynaptics/NEO-TX) (Shadow Desktop).

## Design Principles

- **Local-first.** Everything runs on your hardware. No cloud dependency.
- **Model-aware.** Knows which models are loaded, their VRAM/RAM cost, and routes accordingly.
- **Thin.** Under 1000 lines. No bloat. If it's not routing, model management, or API — it doesn't belong here.

## Models

### Active Stack

| Model | Role | Size | Where | Speed | Purpose |
|-------|------|------|-------|-------|---------|
| **UI-TARS-72B** (Q4) | Visuomotor Agent | ~42GB | CPU (128GB RAM) | 3-5 tok/s | GUI interaction — screenshot in, click/type out |
| **Qwen2.5-Coder-14B** | Planner / Reasoning | ~9.4GB | GPU (RTX 4070) | 30-50 tok/s | Intent parsing, task decomposition, code |
| **Qwen3-8B** | Fast Chat | ~5.2GB | GPU (swapped) | 40-60 tok/s | Quick responses, triviality handling |

### Why These Models

**UI-TARS-72B** (ByteDance, open-weight):
- Purpose-built for computer use — trained on GUI interaction trajectories
- 72B on CPU is slow (~3-5 tok/s) but accurate. For GUI agent work, accuracy beats speed — each action output is a short JSON, not a novel
- Competitive with Claude Computer Use on benchmarks
- The 128GB RAM is the moat — most people can't run 72B locally

**Qwen2.5-Coder-14B** (Alibaba, open-weight):
- Best coding model at 14B parameter class
- Fits entirely on RTX 4070 (12GB VRAM)
- Fast enough for interactive use (30-50 tok/s)
- Handles intent parsing, task decomposition, planning

**Qwen3-8B** (Alibaba, open-weight):
- Lightweight chat model for trivial follow-ups
- Swaps in/out of GPU as needed
- Handles "yes", "tell me more", "why?" without burning 14B capacity

### Model Routing Strategy

```
User input arrives
    │
    ├─ Triviality detector (regex, zero LLM cost)
    │   └─ Trivial? → Qwen3-8B (fast, cheap)
    │
    ├─ Needs GUI interaction?
    │   └─ Yes → UI-TARS-72B on CPU (screenshot → action)
    │
    └─ Complex reasoning / planning?
        └─ Qwen2.5-Coder-14B on GPU (fast, accurate)
```

## Architecture

```
┌─────────────────────────────────────────┐
│              Alchemy Core               │
│                                         │
│  ┌─────────┐  ┌──────────┐  ┌───────┐  │
│  │ Router  │  │  Model   │  │  Auth │  │
│  │         │  │ Manager  │  │       │  │
│  │classify │  │ load     │  │bearer │  │
│  │escalate │  │ unload   │  │trust  │  │
│  │gateway  │  │ health   │  │       │  │
│  └────┬────┘  └────┬─────┘  └───────┘  │
│       │            │                    │
│  ┌────▼────────────▼────────────────┐   │
│  │         FastAPI Server           │   │
│  │         port 8000                │   │
│  └──────────────────────────────────┘   │
└──────────────────┬──────────────────────┘
                   │ API
        ┌──────────┼──────────┐
        ▼          ▼          ▼
    NEO-TX      Future      Future
   (Shadow     (Mobile)    (Plugin)
   Desktop)
```

## API Endpoints

```
# Health
GET  /health                    → {"status": "ok", "models": {...}}

# Chat / Routing
POST /chat                      → Route to best model, return response
POST /chat/stream               → Same but streaming

# Model Management
GET  /models                    → List loaded models + VRAM/RAM usage
POST /models/load               → Load a model (GPU or CPU)
POST /models/unload             → Unload a model
GET  /models/health             → Ollama status + per-model stats

# Vision (for NEO-TX)
POST /vision/analyze            → Send screenshot, get action JSON from UI-TARS
```

## Quick Start

```bash
# 1. Clone
git clone https://github.com/NeoSynaptics/Alchemy.git
cd Alchemy

# 2. Install
pip install -e .

# 3. Pull models
ollama pull qwen2.5-coder:14b
ollama pull qwen3:8b
# UI-TARS-72B: ollama pull ui-tars:72b (when available, or manual GGUF import)

# 4. Run
python -m alchemy
# → Server on http://localhost:8000
```

## Hardware Requirements

- **GPU:** RTX 4070 (12GB VRAM) — runs 14B model
- **RAM:** 64GB minimum, 128GB recommended — runs 72B on CPU
- **CPU:** Modern multi-core (i9-13900K or equivalent) — CPU inference for 72B

## Connected Projects

- **[NEO-TX](https://github.com/NeoSynaptics/NEO-TX)** — Shadow Desktop. AI operates a hidden virtual desktop. Connects to Alchemy for model routing and vision analysis.

## License

MIT — NeoSynaptics
