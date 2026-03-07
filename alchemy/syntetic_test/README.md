# AlchemySynteticTest

Desktop app for end-to-end synthetic testing of the Alchemy ecosystem. Open it, press a button, watch the full pipeline get tested on screen.

## Quick Start

```bash
pip install -e .
python app.py
```

Or double-click the desktop shortcut (run `python create_shortcut.py` to create it).

## Test Suites

| Button | What it does |
|--------|-------------|
| **AlchemyVoice** | Cold start model, mic check, synthetic chat input, measure response time, send second message, measure warm time |
| **AlchemyClick** | Browser a11y click + Flow vision coordinate tests (coming) |
| **AlchemyGPU** | VRAM status, model loading, GPU health, timeout diagnostics (coming) |
| **AlchemyMemory** | Memory persistence and recall (not wired) |

## Timeout Protection

If Voice (or any test) takes too long or freezes, it routes to AlchemyGPU test automatically to diagnose whether it's a GPU/VRAM issue.

## Services

| Service | Default URL |
|---------|-------------|
| Alchemy | `http://localhost:8000` |
| AlchemyVoice | `http://localhost:8100` |
