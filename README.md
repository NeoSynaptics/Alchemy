# Alchemy

**Core AI engine -- voice, click automation, GPU orchestration, research browser, local-first.**

Alchemy is a single-server AI engine running on port 8000. It manages GPU/CPU model fleets, runs AlchemyClick for GUI automation (Playwright + vision), AlchemyVoice for speech interaction, and hosts modular AI apps. Everything runs on your hardware.

## Design Principles

- **Local-first.** Everything runs on your hardware. No cloud dependency.
- **Monorepo.** Voice, click, GPU orchestration, research -- all in one repo, one server.
- **Many small specialists > one slow giant.** ~28GB GPU VRAM across two cards, ~42GB CPU RAM.

## Modules

| Module | Tier | What it does |
|--------|------|--------------|
| **AlchemyVoice** | core | Voice pipeline (Whisper STT, Fish Speech TTS, wake word, smart router, tray) |
| **AlchemyClick** | core | GUI automation (Playwright a11y + Qwen2.5-VL vision, ghost cursor) |
| **GPU Orchestrator** | infra | VRAM/RAM fleet management, model placement, hot-swap |
| **AlchemyBrowser** | app | Web research (DuckDuckGo + trafilatura + Qwen3 synthesis) |
| **Gate Reviewer** | app | Claude Code auto-approve (Qwen3 14B review) |
| **Desktop Agent** | app | Native Windows automation (ghost cursor, SendInput) |

## Models

| Model | Hardware | Role |
|-------|----------|------|
| Qwen3 14B | GPU 1 (16GB) | Conversational, routing, gate review |
| Qwen2.5-VL 7B | GPU 1 (16GB) | Vision agent, escalation fallback |
| Whisper large-v3 | GPU 0 (12GB) | Speech-to-text |
| Fish Speech S1 | GPU 0 (12GB) | Text-to-speech |
| UI-TARS-72B Q4_K_M | CPU (128GB RAM) | GUI visuomotor agent |

## Quick Start

```bash
# 1. Clone
git clone https://github.com/NeoSynaptics/Alchemy.git
cd Alchemy

# 2. Install
pip install -e ".[voice]"

# 3. Run
make server              # -> http://localhost:8000
```

## API

```
GET  /health                        -> server status
POST /v1/vision/task                -> submit multi-step GUI task
POST /v1/vision/analyze             -> screenshot -> action JSON
POST /v1/chat                       -> non-streaming chat
POST /v1/chat/stream                -> SSE streaming chat
GET  /v1/voice/status               -> voice pipeline state
POST /v1/voice/start|stop|mode      -> control voice
GET  /v1/modules                    -> module discovery
GET  /v1/gpu/status                 -> GPU/RAM status
```

## Hardware

- **GPU:** RTX 5060 Ti 16GB + RTX 4070 12GB (28GB total VRAM)
- **CPU:** i9-13900K
- **RAM:** 128GB

## Connected Projects

- **[NEO-RX](https://github.com/NeoSynaptics/NEO-RX)** -- Temporal constitution + Nightwatch

## License

MIT -- NeoSynaptics
