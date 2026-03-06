# Alchemy Architecture

## Overview

Alchemy is the core AI engine. Everything runs in a single server on port 8000: voice pipeline, click automation (shadow desktop + Playwright), GPU orchestration, research browser, and more.

```
YOUR PC (i9-13900K, 128GB RAM, RTX 4070 12GB + RTX 5060 Ti 16GB)
+---------------------------------------------------------------+
|                                                                 |
|  WINDOWS 11 (your screen -- untouched)                         |
|                                                                 |
|  +---------------------------+  +----------------------------+  |
|  |  Alchemy (port 8000)      |  |  Ollama (port 11434)       |  |
|  |  Core AI engine           |  |                            |  |
|  |                           |  |  GPU 1 (16GB):             |  |
|  |  - AlchemyVoice           |  |    Qwen3 14B (9GB)         |  |
|  |    (voice, chat, tray)    |  |    Qwen2.5-VL 7B (4.4GB)   |  |
|  |  - AlchemyClick           |  |                            |  |
|  |    (GUI automation)       |  |  GPU 0 (12GB):             |  |
|  |  - Shadow desktop ctrl    |  |    Whisper (1GB)           |  |
|  |  - GPU orchestrator       |  |    Fish Speech (5GB)       |  |
|  |  - Research browser       |  |                            |  |
|  |  - Gate reviewer          |  |  CPU (128GB RAM):          |  |
|  |                           |  |    UI-TARS-72B (~42GB)     |  |
|  +---------------------------+  +----------------------------+  |
|                                                                 |
|  +---------------------------+  +----------------------------+  |
|  |  System Tray (optional)   |  |  WSL2 Ubuntu               |  |
|  |  PyQt6 widget             |  |                            |  |
|  |  - Approval dialogs       |  |  Xvfb :99 (invisible)     |  |
|  |  - noVNC viewport         |  |  Fluxbox (window manager)  |  |
|  |  - Voice status           |  |  x11vnc -> noVNC (:6080)   |  |
|  +---------------------------+  |  Firefox, LibreOffice...    |  |
|                                 |                            |  |
|                                 |  THE SHADOW DESKTOP        |  |
|                                 |  (controlled by Alchemy)   |  |
|                                 +----------------------------+  |
+---------------------------------------------------------------+
```

## Module Layout

**Core (Tier 0 -- locked):**
- `alchemy/core/` -- Playwright agent kernel, browser manager, approval gate
- `alchemy/click/` -- AlchemyClick (Playwright + vision-based GUI automation)
- `alchemy/voice/` -- AlchemyVoice (voice pipeline, smart router, conversation, tray)
- `alchemy/cloud/` -- Cloud AI Bridge (provider-agnostic, Claude primary)

**Infrastructure (Tier 1):**
- `alchemy/gpu/` -- VRAM/RAM fleet management, model placement
- `alchemy/adapters/` -- LLM adapters (Ollama, vLLM, GUI-Actor)
- `alchemy/shadow/` -- Shadow desktop (WSL2 bridge, controller)
- `alchemy/router/` -- Context router (request classification)

**App (Tier 2 -- freely changeable):**
- `alchemy/desktop/` -- Native Windows automation (ghost cursor)
- `alchemy/gate/` -- Gate reviewer (Claude Code auto-approve)
- `alchemy/research/` -- AlchemyBrowser (web research)
- `alchemy/word/` -- AlchemyWord (text editor suggestions)

## Resource Split

```
GPU 1 (RTX 5060 Ti, 16GB VRAM):    GPU 0 (RTX 4070, 12GB VRAM):
  Qwen3 14B       = ~9GB (resident)    Whisper large-v3 = ~1GB (resident)
  Qwen2.5-VL 7B   = ~4.4GB (resident)  Fish Speech      = ~5GB (resident)

CPU (i9-13900K, 128GB RAM):
  UI-TARS-72B Q4_K_M = ~42GB
  Remaining          = ~86GB free
```

GPU and CPU never block each other. They work in parallel.

## Click Agent Loop

```
Voice/API sends: POST /v1/vision/task {goal: "send email with hours"}
    |
    v
Alchemy agent loop:
    1. Capture screenshot from Xvfb (:99)
    2. Send to vision model -> get action JSON
    3. Classify action tier (AUTO / NOTIFY / APPROVE)
    4. If APPROVE -> pause, show approval dialog via tray
    5. Execute action via xdotool in WSL2
    6. Repeat (max 50 steps)
    |
    v
Result: task complete / approval request / status update
```
