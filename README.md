# Alchemy

**Local-first LLM core engine. Model management, routing, voice pipeline, and shared API.**

Alchemy is the foundation layer that any Alchemy-ecosystem tool connects to. It manages local models via Ollama, routes requests to the right model, handles voice input/output, and exposes a clean API. Currently serves [NEO-TX](https://github.com/NeoSynaptics/NEO-TX) (Shadow Desktop).

## Design Principles

- **Local-first.** Everything runs on your hardware. No cloud dependency.
- **Model-aware.** Knows which models are loaded, their VRAM/RAM cost, and routes accordingly.
- **Thin.** No bloat. If it's not routing, model management, voice, or API — it doesn't belong here.

## Models

### Active Stack

| Model | Role | Size | Where | Speed | Purpose |
|-------|------|------|-------|-------|---------|
| **UI-TARS-72B** (Q4) | Visuomotor Agent | ~42GB | CPU (128GB RAM) | 3-5 tok/s | GUI interaction — screenshot in, click/type out |
| **Qwen2.5-Coder-14B** | Planner / Reasoning | ~9.4GB | GPU (RTX 4070) | 30-50 tok/s | Intent parsing, task decomposition, voice interpretation |
| **Qwen3-8B** | Fast Chat | ~5.2GB | GPU (swapped) | 40-60 tok/s | Quick responses, triviality handling |

### Model Routing Strategy

```
User input arrives (text or voice)
    │
    ├─ Voice? → Whisper STT → text
    │
    ├─ Triviality detector (regex, zero LLM cost)
    │   └─ Trivial? → Qwen3-8B (fast, cheap)
    │
    ├─ Needs GUI interaction?
    │   └─ Yes → route to NEO-TX → UI-TARS-72B on CPU
    │
    └─ Complex reasoning / planning?
        └─ Qwen2.5-Coder-14B on GPU (fast, accurate)
```

### Future: Adapter Pattern (Apple-inspired)

One base model stays resident (~9GB for 14B). Tiny LoRA adapters hot-swap per request (~200MB each, 1-5ms switch):
- **Routing classifier** — replaces regex with actual understanding
- **Code understanding** — structured diff analysis
- **Doc classification** — fast categorization
- **Intent parser** — natural language → structured task spec

Requires llama.cpp server (Ollama doesn't support LoRA hot-swap yet). Train with Unsloth.

## Architecture

```
┌───────────────────────────────────────────────┐
│                 Alchemy Core                  │
│                 port 8000                     │
│                                               │
│  ┌─────────┐  ┌──────────┐  ┌─────────────┐  │
│  │ Router  │  │  Model   │  │    Voice    │  │
│  │         │  │ Manager  │  │             │  │
│  │classify │  │ load     │  │ wake word   │  │
│  │escalate │  │ unload   │  │ STT (Whisp) │  │
│  │gateway  │  │ health   │  │ TTS (Piper) │  │
│  └────┬────┘  └────┬─────┘  └──────┬──────┘  │
│       │            │               │          │
│  ┌────▼────────────▼───────────────▼──────┐   │
│  │            FastAPI Server              │   │
│  │            port 8000                   │   │
│  └────────────────────────────────────────┘   │
│       │                                       │
│  ┌────▼──────────────────────────────────┐    │
│  │         Ollama (localhost:11434)       │    │
│  │                                       │    │
│  │  GPU: Qwen2.5-Coder-14B (resident)   │    │
│  │  GPU: Qwen3-8B (swapped)             │    │
│  │  CPU: UI-TARS-72B (128GB RAM)         │    │
│  └───────────────────────────────────────┘    │
└──────────────────┬────────────────────────────┘
                   │ HTTP API
        ┌──────────┼──────────┐
        ▼          ▼          ▼
    NEO-TX      Future      Future
   (Shadow     (Mobile)    (Plugin)
   Desktop)
```

### Voice Flow

```
Mic → openWakeWord ("Hey Neo") → faster-whisper (STT, GPU)
  → 14B interprets intent
    ├─ Needs GUI? → route to NEO-TX shadow desktop
    └─ Text answer? → Piper TTS (CPU) → speaker
```

Voice lives in Alchemy because it's a general input/output layer. NEO-TX never touches audio — it receives pre-parsed intent via API.

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

# Voice
POST /voice/transcribe          → Audio → text (Whisper)
POST /voice/speak               → Text → audio (Piper TTS)
GET  /voice/status              → Voice pipeline health
```

## Quick Start

```bash
# 1. Clone
git clone https://github.com/NeoSynaptics/Alchemy.git
cd Alchemy

# 2. Install
pip install -e .

# 3. With voice support
pip install -e ".[voice]"

# 4. Pull models
ollama pull qwen2.5-coder:14b
ollama pull qwen3:8b
# UI-TARS-72B: ollama pull ui-tars:72b (when available, or manual GGUF import)

# 5. Run
python -m alchemy
# → Server on http://localhost:8000
```

## Hardware Requirements

- **GPU:** RTX 4070 (12GB VRAM) — runs 14B model + Whisper STT
- **RAM:** 64GB minimum, 128GB recommended — runs 72B on CPU
- **CPU:** Modern multi-core (i9-13900K or equivalent) — CPU inference for 72B + Piper TTS

## Connected Projects

- **[NEO-TX](https://github.com/NeoSynaptics/NEO-TX)** — Shadow Desktop. AI operates a hidden virtual desktop. Connects to Alchemy for model routing and vision analysis.

## License

MIT — NeoSynaptics
