# Alchemy Model Stack

## Active Models

### UI-TARS-72B — GUI Visuomotor Agent (CPU)
- **Source:** ByteDance (open-weight)
- **Role:** Model B in NEO-TX — takes screenshots, outputs mouse/keyboard actions
- **Quantization:** Q4_K_M (~42GB RAM)
- **Runs on:** CPU (i9-13900K, 128GB RAM)
- **Speed:** ~3-5 tok/s (fine — outputs are short action JSONs)
- **Why 72B on CPU?** Accuracy > speed for GUI tasks. Each action is a short JSON like `{"action": "click", "x": 340, "y": 200}`. 3 seconds of thinking per click is acceptable. The 7B version misses too many UI elements.
- **Benchmarks:** Competitive with Claude Computer Use (~50%+ on GUI benchmarks)
- **Pull:** `ollama pull ui-tars:72b` (or manual GGUF import)

### Qwen2.5-Coder-14B — Planner / Reasoning (GPU)
- **Source:** Alibaba (open-weight)
- **Role:** Model A — intent parsing, task decomposition, code generation
- **VRAM:** ~9.4GB on RTX 4070
- **Speed:** 30-50 tok/s
- **Why this?** Best coding/reasoning model at 14B class. Fits entirely in 12GB VRAM.
- **Pull:** `ollama pull qwen2.5-coder:14b`

### Qwen3-8B — Fast Chat (GPU, swapped)
- **Source:** Alibaba (open-weight)
- **Role:** Handle trivial follow-ups ("yes", "tell me more", "thanks")
- **VRAM:** ~5.2GB (swaps with 14B as needed)
- **Speed:** 40-60 tok/s
- **Why?** Don't waste 14B capacity on "ok thanks". Triviality detector routes these here.
- **Pull:** `ollama pull qwen3:8b`

## VRAM Budget (RTX 4070, 12GB)

```
Slot 1 (resident): Qwen2.5-Coder-14B  = 9.4GB
Slot 2 (on-demand): Qwen3-8B          = 5.2GB  (swaps with Slot 1)

CPU (128GB RAM):
  UI-TARS-72B Q4_K_M                   = ~42GB  (always loaded when NEO-TX active)
  Remaining                             = ~86GB free
```

Ollama handles model swapping automatically via `keep_alive`. When 14B is idle for 10min, 8B can load. When NEO-TX sends a vision request, 72B is already resident in RAM.

## Routing Logic

```
Input → Triviality Detector (regex, zero cost)
  │
  ├─ Trivial (greeting, "yes", short follow-up)
  │   └─ Qwen3-8B (fast, GPU)
  │
  ├─ Needs vision/GUI interaction
  │   └─ UI-TARS-72B (slow, CPU, accurate)
  │
  └─ Complex reasoning / planning / code
      └─ Qwen2.5-Coder-14B (fast, GPU)
```

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
- Proved: single model can unify language + pixel-level motor control
- Output format: `<Reason>...</Reason><Act>...</Act>` structured blocks
- **Relevance:** Architecture inspiration. Our Model A/B split mirrors their Reason/Act split.

### Pix2Act (DeepMind, 2023)
- First agent with pixel-only input to outperform humans on GUI tasks
- Proved: DOM/accessibility tree NOT needed — pure pixels sufficient
- **Relevance:** Validates our pixel-based approach.

### OpenAI CUA
- GPT-4o + RL training for GUI interaction
- 38.1% OSWorld, 58.1% WebArena, 87% WebVoyager
- **Relevance:** Benchmark reference point.
