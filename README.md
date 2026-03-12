# Alchemy

**Core AI engine -- voice, click automation, GPU orchestration, research browser, local-first.**

Alchemy is a single-server AI engine running on port 8000. It manages GPU/CPU model fleets, runs AlchemyClick for GUI automation (Playwright + vision), AlchemyVoice for speech interaction, and hosts modular AI apps. Everything runs on your hardware.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   server.py (:8000)                   │
├───────────┬──────────┬──────────┬──────────┬─────────┤
│   Voice   │  Click   │ Desktop  │ Research │  Word   │  features
├───────────┴──────────┴──────────┴──────────┴─────────┤
│   APU (GPU fleet)  │  Router  │  Adapters (Ollama)   │  infra
├────────────────────┴──────────┴──────────────────────┤
│          Security  │  Config  │  Registry            │  core
└──────────────────────────────────────────────────────┘
```

## Modules

| Module | ID | Tier | What it does |
|--------|----|------|--------------|
| **AlchemyVoice** | `voice` | core | Voice pipeline (Whisper STT, Fish Speech TTS, wake word, smart router) |
| **AlchemyClick** | `click` | core | GUI automation (Playwright a11y + Qwen2.5-VL vision, ghost cursor) |
| **APU** | `apu` | infra | VRAM/RAM fleet management, model placement, hot-swap, health guard |
| **Security** | `security` | infra | Bearer token authentication middleware |
| **AlchemyWord** | `word` | app | Text generation (summarize, rewrite, expand, translate) |
| **AlchemyBrowser** | `research` | app | Web research (DuckDuckGo + trafilatura + Qwen3 synthesis) |
| **Gate Reviewer** | `gate` | app | Claude Code auto-approve (Qwen3 14B review) |
| **Desktop Agent** | `desktop` | app | Native Windows automation (ghost cursor, SendInput) |

## Setup

```bash
# 1. Clone
git clone https://github.com/NeoSynaptics/Alchemy.git
cd Alchemy

# 2. Configure
cp .env.example .env
# Edit .env with your settings (Ollama URL, auth token, etc.)

# 3. Install
pip install -e ".[voice]"

# 4. Run
make server              # -> http://localhost:8000
```

### Docker

```bash
docker-compose up        # Starts Alchemy + Ollama with GPU
```

### Development

```bash
make dev                 # Install with dev tools
make test                # Run pytest
make ui-install          # Install React UI dependencies
make ui-build            # Build React dashboard (output: ui/dist/)
make ui-dev              # Run React dev server on :5173
```

## API

```
GET  /health                        -> server status
POST /v1/chat                       -> non-streaming chat
POST /v1/chat/stream                -> SSE streaming chat
GET  /v1/voice/status               -> voice pipeline state
POST /v1/voice/start|stop|mode      -> control voice
POST /v1/vision/task                -> submit multi-step GUI task
POST /v1/vision/analyze             -> screenshot -> action JSON
POST /v1/word/generate              -> text generation
GET  /v1/apu/status                 -> GPU/RAM/model status
GET  /v1/modules                    -> module discovery + contracts
```

## Models

| Model | Hardware | Role |
|-------|----------|------|
| Qwen3 14B | GPU 1 (16GB) | Conversational, routing, gate review, text generation |
| Qwen2.5-VL 7B | GPU 1 (16GB) | Vision agent, escalation fallback |
| Whisper large-v3 | GPU 0 (12GB) | Speech-to-text |
| Fish Speech S1 | GPU 0 (12GB) | Text-to-speech |
| UI-TARS-72B Q4_K_M | CPU (128GB RAM) | GUI visuomotor agent |

## Hardware

- **GPU:** RTX 5060 Ti 16GB + RTX 4070 12GB (28GB total VRAM)
- **CPU:** i9-13900K
- **RAM:** 128GB

## Connected Projects

- **[BaratzaMemory](https://github.com/NeoSynaptics/BaratzaMemory)** -- Permanent long-term memory (PostgreSQL + Qdrant)
- **[NEO-RX](https://github.com/NeoSynaptics/NEO-RX)** -- Temporal constitution + Nightwatch

## License

MIT -- NeoSynaptics
