# Alchemy Roadmap

## Done

### Phase 0 -- API Contract
Schemas, stub endpoints, client, tests.

### Phase 1 -- Shadow Desktop
WSL2 live (Xvfb, Fluxbox, x11vnc, noVNC). Bootstrap script.

### Phase 2 -- Vision Agent
Real UI-TARS inference, agent loop, task manager, approval flow.

### Phase 3 -- Context Router
Environment detection, task categories, recovery nudges, completion criteria, context-aware tiers.

### Phase 4 -- Inference Optimization
Official prompt template, streaming with early stop, retry logic, dual-model routing (1.5-7B fast / 72B full), JPEG screenshots, 720p downscale, adaptive timeouts, coordinate validation.

### Phase 5 -- AlchemyVoice Merge
Voice pipeline, smart router, constitution, planner, tray all merged into `alchemy/voice/`.
Clean VoiceSystem interface hides model internals. Single repo, single test suite.

**Current: 1020+ tests, v0.4.0**

---

## Next

### Phase 6 -- Live Integration Test
**Goal:** First real end-to-end run. Prove the full loop works against the actual shadow desktop.

- [ ] Start shadow desktop, open Firefox
- [ ] Submit a task via `/v1/vision/task`: "open firefox and go to google.com"
- [ ] Watch the agent loop: screenshot -> model -> action -> execute
- [ ] Verify the 1.5-7B fast model works on GPU
- [ ] Verify 72B works on CPU as fallback
- [ ] Log real latency numbers (screenshot capture, inference, total step time)
- [ ] Fix any issues found during live testing
- [ ] Document real-world performance numbers

### Phase 7 -- Voice GUI Settings
- [ ] Build settings page for voice mode, wake word, TTS engine
- [ ] Wire VoiceSystem.set_mode() to GUI
- [ ] Replace VoiceCallbackClient HTTP self-calls with direct internal dispatch

---

## Blocked / Waiting

- **UI-TARS-2 weights** -- paper exists (Sep 2025), weights not released by ByteDance
- **vLLM migration** -- only if Ollama quality ceiling becomes a bottleneck
