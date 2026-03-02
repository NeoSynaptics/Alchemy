# Alchemy Model Stack

## Active Model

### UI-TARS-72B — GUI Visuomotor Agent (CPU)
- **Source:** ByteDance (open-weight)
- **Role:** Vision agent — takes screenshots, outputs mouse/keyboard actions
- **Quantization:** Q4_K_M (~42GB RAM)
- **Runs on:** CPU (i9-13900K, 128GB RAM)
- **Speed:** ~3-5 tok/s (fine — outputs are short action JSONs)
- **Why 72B on CPU?** Accuracy > speed for GUI tasks. Each action is a short JSON like `{"action": "click", "x": 340, "y": 200}`. 3 seconds of thinking per click is acceptable. The 7B version misses too many UI elements.
- **Benchmarks:** Competitive with Claude Computer Use (~50%+ on GUI benchmarks)
- **Pull:** `ollama pull ui-tars:72b` (or manual GGUF import)

## NEO-TX Models (GPU-side, separate repo)

These models are owned by NEO-TX, not Alchemy:

| Model | Role | VRAM | Speed |
|-------|------|------|-------|
| 14B conversational | User interaction, semantic understanding | ~9GB | 30-50 tok/s |
| Whisper large-v3 | Speech-to-text (on-demand) | ~3GB | real-time |
| Small specialized (~2B) | Specific fast tasks (future) | ~2GB | 60+ tok/s |
| Piper TTS | Text-to-speech | CPU (~50MB) | real-time |

## Resource Budget

```
CPU (128GB RAM):
  UI-TARS-72B Q4_K_M             = ~42GB  (always loaded when agent active)
  Remaining                       = ~86GB free

GPU (RTX 4070, 12GB VRAM) — managed by NEO-TX:
  14B conversational (resident)   = ~9GB
  Whisper large-v3 (on-demand)    = ~3GB   (swaps with 14B)
  Small models (on-demand)        = ~2GB   (swaps)
```

GPU and CPU work in parallel — never contend for the same resource.

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
- Fine-tuned Gemini Flash-Lite — NOT open source
- Output format: `<Reason>...</Reason><Act>...</Act>` structured blocks
- **Relevance:** Architecture inspiration. Our Alchemy/NEO-TX split mirrors their Reason/Act split.

### Pix2Act (DeepMind, 2023)
- First agent with pixel-only input to outperform humans on GUI tasks
- Proved: DOM/accessibility tree NOT needed — pure pixels sufficient
- **Relevance:** Validates our pixel-based approach.

## Future: Adapter Architecture

Apple-inspired pattern — one base model resident, tiny LoRA adapters hot-swap:

```
14B base model (resident, NEO-TX GPU)
  ├─ Adapter: Routing classifier (~200MB, 1-5ms swap)
  ├─ Adapter: Intent parser (~200MB)
  ├─ Adapter: Doc classification (~200MB)
  └─ Adapter: ... (up to ~20 specialized adapters)
```

Requires llama.cpp server (Ollama doesn't support LoRA hot-swap yet). Train with Unsloth.
