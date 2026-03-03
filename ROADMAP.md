# Alchemy Roadmap

## Done

### Phase 0 — API Contract
Schemas, stub endpoints, client, tests.

### Phase 1 — Shadow Desktop
WSL2 live (Xvfb, Fluxbox, x11vnc, noVNC). Bootstrap script.

### Phase 2 — Vision Agent
Real UI-TARS inference, agent loop, task manager, approval flow.

### Phase 3 — Context Router
Environment detection, task categories, recovery nudges, completion criteria, context-aware tiers.

### Phase 4 — Inference Optimization
Official prompt template, streaming with early stop, retry logic, dual-model routing (1.5-7B fast / 72B full), JPEG screenshots, 720p downscale, adaptive timeouts, coordinate validation.

**Current: 226 tests, 10 commits, v0.2.0**

---

## Next

### Phase 5 — Live Integration Test
**Goal:** First real end-to-end run. Prove the full loop works against the actual shadow desktop.

- [ ] Start shadow desktop, open Firefox
- [ ] Submit a task via `/vision/task`: "open firefox and go to google.com"
- [ ] Watch the agent loop: screenshot → model → action → execute
- [ ] Verify the 1.5-7B fast model works on GPU
- [ ] Verify 72B works on CPU as fallback
- [ ] Log real latency numbers (screenshot capture, inference, total step time)
- [ ] Fix any issues found during live testing
- [ ] Document real-world performance numbers

### Phase 6 — Schema Sync Automation
- [ ] Single source of truth for schemas (generate one from the other, or shared package)
- [ ] CI check that schemas are in sync between Alchemy and NEO-TX

---

## Blocked / Waiting

- **UI-TARS-2 weights** — paper exists (Sep 2025), weights not released by ByteDance
- **vLLM migration** — only if Ollama quality ceiling becomes a bottleneck
- **GPU upgrade** — for running 1.5-7B + Qwen3 14B simultaneously

---

## Not Alchemy's Job

These belong to NEO-TX:
- User interface (tray, voice, approval dialogs)
- Conversational model (Qwen3 14B)
- Task routing (direct answer vs shadow desktop delegation)
- Voice pipeline (Whisper, Piper, wake word)
