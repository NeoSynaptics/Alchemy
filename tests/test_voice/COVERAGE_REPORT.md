# Voice Test Coverage Report

**Date:** 2026-03-11
**Source files:** 40 (in `alchemy/voice/`)
**Test files:** 26 (in `tests/test_voice/`)

## Summary

65% of source files have unit tests. All audio components are **fully mocked** — no real audio, no real models, no real hardware. The pipeline state machine has basic cycle coverage but lacks error recovery and edge case testing.

## Coverage by Component

| Component | Unit Tests | Integration | Real Audio/Model | Grade |
|-----------|-----------|-------------|-----------------|-------|
| Pipeline State Machine | Basic cycle | None | No | 65% |
| STT (Whisper) | Config/calls | None | Mocked | 30% |
| TTS (Piper/Fish) | HTTP/config | None | Mocked | 35% |
| Wake Word | Threshold | None | Mocked | 25% |
| Smart Router | Intent routing | No failover | Mocked | 40% |
| Conversation Manager | Good | Token budget | N/A | 70% |
| Constitution Rules | Good | No enforcement | N/A | 75% |
| Task Planner | Good | No LLM decomp | N/A | 60% |
| Knowledge Retriever | **None** | None | N/A | 0% |
| Event Reporter | **None** | None | N/A | 0% |
| VoiceSystem (interface) | **None** | Via /voice API | Mocked | 20% |
| Tray UI | Event bus only | No GUI tests | No PyQt6 | 10% |

## Critical Gaps (No Tests Exist)

### 1. VoiceSystem (`interface.py`)
Public API for GUI and external code. Only tested indirectly via `/voice` endpoints. Needs direct unit tests for `start()`, `stop()`, `set_mode()`, double-start guard, rapid mode switches.

### 2. Knowledge Retriever (`knowledge/retriever.py`)
Queries NEO-RX for RAG context. No tests for timeout handling, empty results, graceful degradation when NEO-RX is down.

### 3. Event Reporter (`knowledge/reporter.py`)
Fire-and-forget events to NEO-RX. No tests for event delivery, queue overflow, exception handling.

### 4. Tray UI Components (`tray/dialogs.py`, `tray/viewport.py`, `tray/icon.py`)
PyQt6 dialogs, viewport, and system tray icon have zero test coverage.

## State Machine Gaps

**Tested transitions:**
- IDLE → LISTENING → RECORDING → PROCESSING → SPEAKING → IDLE (happy path)
- Double-start guard
- Conversation ID generation

**NOT tested:**
- Stop during RECORDING (cleanup?)
- Router timeout/failure during PROCESSING
- TTS failure during SPEAKING (state recovery?)
- Audio device disconnection during LISTENING
- Rapid mode switches (MUTED during PROCESSING)
- Exception recovery to IDLE
- Empty/very short transcription filtering

## Audio Testing

All audio tests use `np.zeros()` silence or mocked model outputs. No tests with:
- Real speech recordings
- Background noise robustness
- Accent/language variation
- False positive wake word detection
- Audio quality degradation

## Smart Router Gaps

- No real model calls (all providers mocked)
- No provider failure/fallback testing (Ollama down → Alchemy?)
- No conversation continuity testing (multi-turn context preservation)
- No ambiguous intent handling

## Suggested Tests

### P0 — Must Have
1. **`test_interface.py`** — VoiceSystem start/stop/mode direct tests
2. **`test_knowledge_retriever.py`** — graceful degradation, timeout, empty results
3. **`test_knowledge_reporter.py`** — event delivery, exception handling
4. **Expand `test_pipeline.py`** — error recovery, stop-during-recording, TTS failure

### P1 — Important
5. End-to-end audio pipeline (mark `@pytest.mark.slow`, needs GPU)
6. SmartRouter with real 14B model (needs GPU)
7. Provider fallback chain (mock Ollama timeout → Alchemy)
8. Conversation context window stress test (100+ turns)

### P2 — Nice to Have
9. Concurrent `/voice/start` calls (state isolation)
10. Tray dialog tests (using `pytest-qt`)
11. VRAM manager stress test (rapid model cycling)
12. Long-running conversation memory leak detection
