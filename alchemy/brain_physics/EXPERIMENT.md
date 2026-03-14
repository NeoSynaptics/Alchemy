# BrainPhysics — Experiment Design

## Thesis

The brain doesn't brute-force. It:
1. Sketches a crude picture (coarse-to-fine perception)
2. Orients objects spatially ("upper-right = close button") (embodied cognition)
3. Runs cheap physics simulation ("if I click that, Chrome opens") (intuitive physics)
4. Checks if prediction matched reality (predictive processing)
5. Refines only where wrong (error minimization loop)
6. Distills successful patterns for cheap future reuse (consolidation)

**Hypothesis:** A router built on these principles will make better decisions with less compute than flat model routing, because it avoids processing what it already knows and focuses compute on surprises.

---

## Phase 1 — Scene Graph Extraction (Week 1-2)

**Goal:** Given a screenshot, extract a spatial scene graph.

**Input:** Screenshot (1280x720 JPEG)
**Output:** SceneGraph with nodes (objects + bounding boxes) and relations (spatial)

**Method:**
- Pass screenshot to Qwen2.5-VL 7B with prompt:
  "List all interactive objects. For each: label, type, bounding box (x1,y1,x2,y2 normalized), and spatial relation to nearest object."
- Parse response into SceneGraph data structure
- Two passes: coarse (320px) → fine (1280px, only on ambiguous regions)

**Success metric:** >80% of clickable objects detected on 20 test screenshots.

**Compute:** ~2s per screenshot on GPU 1 (qwen2.5vl:7b already loaded).

---

## Phase 2 — Spatial Relation Engine (Week 2-3)

**Goal:** Automatically compute spatial relations between scene nodes.

**Method:**
- From bounding boxes, compute: above/below/left-of/right-of/inside/near/overlaps
- Use physics-style heuristics: distance, containment, alignment
- This is pure geometry — no model needed, CPU-only, <1ms

**Success metric:** Relations match human annotation on 90%+ of object pairs.

---

## Phase 3 — Intuitive Physics Simulation (Week 3-4)

**Goal:** Given a scene graph + candidate action, predict the outcome.

**Input:** SceneGraph + SimAction (e.g., "click node:chrome_icon")
**Output:** SimResult with predicted_outcome + confidence

**Method:**
- Use Qwen3 14B with scene graph as context + action description
- Prompt: "Given this scene layout and the action 'click chrome icon', what happens next? Be brief."
- Compare prediction quality vs. no-scene-graph baseline (just "click at x,y on screenshot")

**Success metric:** Predictions correct >70% of the time. Baseline (no scene graph) should be lower.

**Compute:** ~1s per simulation on GPU 1.

---

## Phase 4 — Prediction Error Loop (Week 4-5)

**Goal:** After performing an action, compare predicted vs actual scene. Refine if wrong.

**Method:**
1. Run Phase 3 simulation → get prediction
2. Execute action (via AlchemyFlow/AlchemyClick)
3. Take new screenshot → run Phase 1 again
4. Compare predicted scene vs actual scene (VLM comparison prompt)
5. If error > threshold: zoom into error region, re-perceive at fine resolution, loop

**Success metric:** Prediction errors decrease across iterations (converges within 3 loops).

---

## Phase 5 — Memory Consolidation (Week 5-6)

**Goal:** Successful patterns get distilled into fast lookup.

**Method:**
- After a successful prediction (error < threshold), store:
  - Scene fingerprint (embedding of scene graph)
  - Action taken
  - Outcome observed
- On future scenes, check memory BEFORE running physics sim
- If similar scene found with high confidence → skip simulation, use cached pattern

**Success metric:** Cache hit rate >30% after 50 interactions. Avg step time decreases.

**Compute:** Embedding lookup ~10ms (nomic-embed-text, already loaded).

---

## Phase 6 — Benchmark (Week 6-8)

**Goal:** Compare BrainPhysics routing vs. current flat routing.

**Scenarios:**
1. "Open Chrome and search for X" — multi-step GUI task
2. "Find the settings button" — spatial reasoning
3. "Close all windows except VS Code" — scene understanding + multiple actions
4. Random desktop screenshots — measure scene understanding accuracy

**Metrics:**
- Task completion rate
- Avg compute per step (ms)
- Number of VLM calls per task
- Prediction accuracy over time (learning curve)

---

## Architecture (How It Fits in Alchemy)

```
                    ┌─────────────────┐
                    │  BrainPhysics   │  ← THIS MODULE
                    │   Engine        │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Perceive │  │ Simulate │  │ Distill  │
        │ (VLM)    │  │ (LLM)   │  │ (Embed)  │
        └────┬─────┘  └────┬─────┘  └────┬─────┘
             │              │              │
             ▼              ▼              ▼
        ┌──────────────────────────────────────┐
        │         APU Gateway                  │  ← existing infra
        │  (qwen2.5vl:7b, qwen3:14b, nomic)  │
        └──────────────────────────────────────┘
```

**No new models needed.** Uses existing fleet:
- Qwen2.5-VL 7B (GPU 1, already loaded) — perception
- Qwen3 14B (GPU 1, already loaded) — simulation/reasoning
- nomic-embed-text (RAM, already loaded) — consolidation lookup

**No new hardware needed.** Runs within current VRAM budget.

---

## Key Insight

This is NOT "yet another agent loop." The difference:

| Traditional Agent | BrainPhysics |
|---|---|
| Full screenshot → VLM every step | Coarse first, fine only where needed |
| No spatial model | Explicit scene graph with physics relations |
| No prediction | Predicts before acting, checks after |
| No memory | Distills patterns, gets faster over time |
| Fixed compute per step | Adaptive compute — easy tasks = cheap, hard = more loops |

The brain is efficient because it **doesn't process everything** — it predicts most of it and only computes surprises. That's the core idea.
