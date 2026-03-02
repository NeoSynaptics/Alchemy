# Alchemy Architecture

## Overview

Alchemy is the core engine. It manages local LLM models and exposes a routing API. Client tools (NEO-TX, future tools) connect via HTTP.

```
┌─────────────────────────────────────┐
│           Alchemy Core              │
│           port 8000                 │
│                                     │
│  ┌───────────┐  ┌───────────────┐   │
│  │  Router   │  │ Model Manager │   │
│  │           │  │               │   │
│  │ classify  │  │ Ollama API    │   │
│  │ escalate  │  │ load/unload   │   │
│  │ gateway   │  │ health/stats  │   │
│  └─────┬─────┘  └───────┬───────┘   │
│        │                │           │
│  ┌─────▼────────────────▼────────┐  │
│  │       Ollama (localhost:11434) │  │
│  │                               │  │
│  │  GPU: Qwen2.5-Coder-14B      │  │
│  │  GPU: Qwen3-8B (swapped)     │  │
│  │  CPU: UI-TARS-72B            │  │
│  └───────────────────────────────┘  │
└──────────────────┬──────────────────┘
                   │ HTTP API
        ┌──────────┼──────────┐
        ▼          ▼          ▼
    NEO-TX      Future      Future
```

## Separation of Concerns

**Alchemy** (this repo):
- Ollama model lifecycle (load, unload, health)
- Request routing (which model handles which request)
- Shared API server (port 8000)
- Auth (bearer tokens)

**NEO-TX** (separate repo):
- Shadow desktop (WSL2 + Xvfb)
- Agent loop (screenshot → action)
- Tray widget + viewport
- Voice interface
- Defense constitution (approval gates)

**The boundary is the API.** NEO-TX calls `POST /chat` and `POST /vision/analyze`. It never touches Ollama directly.
