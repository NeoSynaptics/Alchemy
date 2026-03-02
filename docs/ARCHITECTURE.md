# Alchemy Architecture

## Overview

Alchemy is the CPU-side core engine. It runs UI-TARS-72B for GUI interaction on a hidden shadow desktop. NEO-TX (GPU-side, user-facing) delegates heavy GUI work here.

```
YOUR PC (i9-13900K, 128GB RAM, RTX 4070 12GB)
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  WINDOWS 11 (your screen — untouched)                          │
│                                                                 │
│  ┌─────────────────────────┐  ┌──────────────────────────────┐  │
│  │  Alchemy (port 8000)    │  │  Ollama (port 11434)         │  │
│  │  CPU-side core engine   │  │                              │  │
│  │                         │  │  CPU (128GB RAM):            │  │
│  │  - Shadow desktop ctrl  │  │    UI-TARS-72B (~42GB)       │  │
│  │  - Vision agent loop    │  │                              │  │
│  │  - /vision/analyze      │  │  GPU (12GB VRAM):            │  │
│  │  - /shadow/start|stop   │  │    14B conversational (NEO)  │  │
│  │                         │  │    + small models (NEO)      │  │
│  └─────────────────────────┘  │    + Whisper STT (NEO)       │  │
│                                │                              │  │
│  ┌─────────────────────────┐  └──────────────────────────────┘  │
│  │  NEO-TX (port 8100)     │                                    │
│  │  GPU-side smart iface   │  ┌────────────────────────────┐    │
│  │                         │  │  WSL2 Ubuntu               │    │
│  │  - Voice pipeline       │  │                            │    │
│  │  - 14B conversation     │  │  Xvfb :99 (invisible)     │    │
│  │  - Tray widget          │  │  Fluxbox (window manager)  │    │
│  │  - Approval gates       │  │  x11vnc → noVNC (:6080)   │    │
│  │  - Small specialized    │  │  Firefox, LibreOffice...   │    │
│  │    GPU models           │  │                            │    │
│  │                         │  │  THE SHADOW DESKTOP        │    │
│  │  Delegates GUI tasks ───┼──┤  (controlled by Alchemy)   │    │
│  │  to Alchemy             │  │                            │    │
│  └─────────────────────────┘  └────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Separation of Concerns

**Alchemy** (this repo) — CPU side:
- Shadow desktop (WSL2 + Xvfb + Fluxbox + x11vnc + noVNC)
- Vision agent loop (screenshot → UI-TARS-72B → action → xdotool)
- CPU model lifecycle (UI-TARS-72B load/unload/health)
- API server (port 8000)
- Auth (bearer tokens)

**NEO-TX** (separate repo) — GPU side:
- Voice pipeline (Whisper STT + Piper TTS + wake word)
- 14B conversational model (semantic, NOT coding)
- Small specialized GPU models for specific fast tasks
- Tray widget + viewport (noVNC view into shadow desktop)
- Defense constitution (approval gates)
- User interaction layer

**The boundary is the API.** NEO-TX calls Alchemy endpoints for GUI tasks. Alchemy calls NEO-TX back for approval on dangerous actions.

## Resource Split

```
CPU (i9-13900K, 128GB RAM):           GPU (RTX 4070, 12GB VRAM):
  UI-TARS-72B Q4_K_M  = ~42GB           14B conversational  = ~9GB (resident)
  Piper TTS            = ~50MB           Whisper large-v3    = ~3GB (on-demand)
  Shadow desktop       = minimal         Small models        = ~2GB (on-demand)
  Remaining            = ~86GB free      Scheduling: time-share VRAM
```

GPU and CPU never block each other. They work in parallel.

## Agent Loop (Alchemy-side)

```
NEO-TX sends: POST /vision/task {goal: "send email with hours"}
    │
    ▼
Alchemy agent loop:
    1. Capture screenshot from Xvfb (:99)
    2. Send to UI-TARS-72B → get action JSON
    3. Classify action tier (AUTO / NOTIFY / APPROVE)
    4. If APPROVE → pause, request approval from NEO-TX
    5. Execute action via xdotool in WSL2
    6. Repeat (max 50 steps)
    │
    ▼
NEO-TX receives: task complete / approval request / status update
```
