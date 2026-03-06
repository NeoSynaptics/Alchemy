# Alchemy Model Stack

## Active Models

### GUI Stack (GPU 1 — RTX 5060 Ti 16GB)
| Model | Role | VRAM | Speed |
|-------|------|------|-------|
| Qwen3 14B | Conversational, routing, gate review | ~9GB | 30-50 tok/s |
| Qwen2.5-VL 7B | Vision agent (AlchemyFlow), escalation | ~4.4GB | real-time |

### Voice Stack (GPU 0 — RTX 4070 12GB)
| Model | Role | VRAM | Speed |
|-------|------|------|-------|
| Whisper large-v3 | Speech-to-text | ~1GB | real-time |
| Fish Speech S1 | Text-to-speech | ~5GB | real-time |

### CPU Stack (128GB RAM)
| Model | Role | RAM | Speed |
|-------|------|-----|-------|
| UI-TARS-72B Q4_K_M | GUI visuomotor agent | ~42GB | 3-5 tok/s |
| Piper TTS | Fallback text-to-speech | ~50MB | real-time |

## Resource Budget

```
GPU 1 (RTX 5060 Ti, 16GB VRAM):
  Qwen3 14B (resident)             = ~9GB
  Qwen2.5-VL 7B (resident)         = ~4.4GB

GPU 0 (RTX 4070, 12GB VRAM):
  Whisper large-v3 (resident)       = ~1GB
  Fish Speech S1 (resident)         = ~5GB

CPU (128GB RAM):
  UI-TARS-72B Q4_K_M               = ~42GB  (always loaded when agent active)
  Remaining                         = ~86GB free
```

GPU and CPU work in parallel -- never contend for the same resource.

## Alternative Models (Evaluated)

| Model | Why considered | Why not chosen |
|-------|---------------|----------------|
| CogAgent-2 (18B) | GUI-trained | Smaller than UI-TARS, less accurate |
| SeeClick (9.6B) | UI grounding | Specialist only, can't plan |
| ShowUI (2B/7B) | Lightweight | Too small for reliable GUI work |
| Llama 3.2 Vision (11B) | General vision | Not GUI-trained, poor at clicking |
| Claude Computer Use | Production-grade | Cloud-only, costs money, privacy concern |

## Prior Art

### SIMA 2 (DeepMind, 2024)
- 62% task completion on gaming environments
- Fine-tuned Gemini Flash-Lite -- NOT open source
- Output format: `<Reason>...</Reason><Act>...</Act>` structured blocks
- **Relevance:** Architecture inspiration. Reason/Act split mirrors our voice router + click agent.

### Pix2Act (DeepMind, 2023)
- First agent with pixel-only input to outperform humans on GUI tasks
- Proved: DOM/accessibility tree NOT needed -- pure pixels sufficient
- **Relevance:** Validates our pixel-based approach in AlchemyFlow.

## Future: Adapter Architecture

Apple-inspired pattern -- one base model resident, tiny LoRA adapters hot-swap:

```
14B base model (resident, GPU 1)
  +- Adapter: Routing classifier (~200MB, 1-5ms swap)
  +- Adapter: Intent parser (~200MB)
  +- Adapter: Doc classification (~200MB)
  +- Adapter: ... (up to ~20 specialized adapters)
```

Requires llama.cpp server (Ollama doesn't support LoRA hot-swap yet). Train with Unsloth.
