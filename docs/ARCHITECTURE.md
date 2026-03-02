# Alchemy Architecture

## Overview

Alchemy is the core engine. It manages local LLM models, handles voice I/O, and exposes a routing API. Client tools (NEO-TX, future tools) connect via HTTP.

```
┌───────────────────────────────────────────────┐
│                 Alchemy Core                  │
│                 port 8000                     │
│                                               │
│  ┌─────────┐  ┌──────────┐  ┌─────────────┐  │
│  │ Router  │  │  Model   │  │    Voice    │  │
│  │         │  │ Manager  │  │             │  │
│  │classify │  │ Ollama   │  │ wake word   │  │
│  │escalate │  │ load     │  │ STT (Whisp) │  │
│  │gateway  │  │ unload   │  │ TTS (Piper) │  │
│  └────┬────┘  └────┬─────┘  └──────┬──────┘  │
│       │            │               │          │
│  ┌────▼────────────▼───────────────▼──────┐   │
│  │         Ollama (localhost:11434)        │   │
│  │                                        │   │
│  │  GPU: Qwen2.5-Coder-14B (resident)    │   │
│  │  GPU: Qwen3-8B (swapped)              │   │
│  │  CPU: UI-TARS-72B                      │   │
│  └────────────────────────────────────────┘   │
└──────────────────┬────────────────────────────┘
                   │ HTTP API
        ┌──────────┼──────────┐
        ▼          ▼          ▼
    NEO-TX      Future      Future
```

## Separation of Concerns

**Alchemy** (this repo):
- Ollama model lifecycle (load, unload, health)
- Request routing (which model handles which request)
- Voice pipeline (wake word, STT, TTS, intent routing)
- Shared API server (port 8000)
- Auth (bearer tokens)

**NEO-TX** (separate repo):
- Shadow desktop (WSL2 + Xvfb)
- Agent loop (screenshot → action via UI-TARS)
- Tray widget + viewport
- Defense constitution (approval gates)

**The boundary is the API.** NEO-TX calls `POST /chat` and `POST /vision/analyze`. It never touches Ollama directly. It never touches audio — Alchemy handles voice and sends pre-parsed intent.

## Voice Flow

```
Mic (Windows)
  → openWakeWord ("Hey Neo", CPU, ~10MB)
  → faster-whisper STT (GPU, on-demand ~3GB VRAM)
  → Qwen2.5-Coder-14B interprets intent (GPU, resident)
    ├─ Needs GUI? → POST to NEO-TX /task (shadow desktop handles it)
    └─ Text answer? → Piper TTS (CPU, ~50MB) → speaker
```

## Future: Adapter Architecture

Apple-inspired pattern — one base model resident, tiny LoRA adapters hot-swap:

```
Qwen2.5-Coder-14B (base, resident ~9GB VRAM)
  ├─ Adapter: Routing classifier (~200MB, 1-5ms swap)
  ├─ Adapter: Code understanding (~200MB)
  ├─ Adapter: Doc classification (~200MB)
  └─ Adapter: Intent parser (~200MB)
```

Requires llama.cpp server (Ollama doesn't support LoRA hot-swap yet). Potential replacement for regex-based triviality detection with actual learned routing.
