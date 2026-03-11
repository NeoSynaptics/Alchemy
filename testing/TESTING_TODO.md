# Testing TODO

Distilled from test results. Claude reads this, fixes issues, re-runs tests.

---

## Alchemy

**Baseline (2026-03-11):** 1104 passed, 13 failed

### Known Failures
1. **5 timing mock issues** — `test_voice/` tests with `asyncio.sleep` mocking fragility
2. **8 missing `duckduckgo_search` dep** — `test_research/` needs the package installed (PC only)

### Action Items
- [ ] Fix voice timing mocks (laptop-safe, no GPU needed)
- [ ] Install `duckduckgo_search` on PC and re-run research tests

---

## NEOSY

**Baseline (2026-03-11):** 42 passed, 0 failed

### Known Failures
None — all tests pass with mocked infrastructure.

### Action Items
- [ ] Review tests for core coverage (trim redundant, keep essential)
- [ ] Run against real Docker + Qdrant on PC (Task 24)

---

## Future: AI Personality Tests

Beyond system correctness, test NEO's behavior as an entity:

- [ ] **Classification quality** — Does NEO's Qwen output match expected topic/entity extraction?
- [ ] **Thought coherence** — Are NEO's generated thoughts logically connected to source memories?
- [ ] **Voice routing accuracy** — Does the smart router correctly distinguish conversation vs GUI task vs command?
- [ ] **Personality consistency** — Does NEO maintain consistent tone/style across interactions?
- [ ] **Reaction appropriateness** — Does NEO's sentiment analysis of user reactions make sense?
- [ ] **Research suggestions** — Are NEO's suggested external resources relevant and safe?

These require curated test cases with expected outputs — not just pass/fail assertions.

---

## How to Update This File

1. Run `bash testing/run_tests.sh`
2. Read results in `testing/results/`
3. Update this file with new failures/fixes
4. Commit and push
