# AlchemyOS GUI Agent — Playwright Build Plan

## The Core Insight

Everyone else treats GUI automation as a **vision problem** (screenshot → VLM → coordinates → click). That's wrong for web and Electron apps. It's a **structured data problem**. The DOM and accessibility tree already contain everything a model needs. Better input data means smaller models work better than bigger models with worse input.

## Architecture: Three Components

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│ Playwright   │────▶│  Qwen3 14B       │────▶│ Playwright   │
│ (snapshot)   │     │  (think: true)   │     │ (execute)    │
│              │     │                  │     │              │
│ Accessibility│     │  Reads tree +    │     │  click @ref  │
│ tree + DOM   │     │  action log,     │     │  type @ref   │
│ with refs    │     │  outputs action  │     │  scroll      │
└─────────────┘     └──────────────────┘     └─────────────┘
       ▲                                            │
       └────────────────────────────────────────────┘
                    (loop until task done)
```

No vision model. No screenshots. No coordinate grounding. No cloud API.

## Tiered Architecture

### Tier 1: Playwright Agent (PRIMARY)
- Accessibility tree → Qwen3 14B (think: true) → ref-based actions
- Targets: Chrome, VS Code, Spotify, Slack, Discord, Notion, Office 365 (web)
- Speed: 2-5 sec per action decision (deep reasoning, background agent)
- Reliability: Deterministic ref-based clicks, no pixel guessing

### Tier 2: UI-TARS Vision Agent (FALLBACK)
- Shadow desktop screenshot → UI-TARS 72B → coordinate-based actions
- Targets: Native Win32 apps, anything without DOM/accessibility tree
- Existing code preserved in alchemy/vision/ and alchemy/shadow/

## Hardware

- **GPU (RTX 4070, 12GB):** Qwen3 14B — already running via Ollama, warm in memory
- **GPU (RTX 5060 Ti, 16GB):** Available for voice pipeline + other tasks (incoming)
- **CPU (i9-13900K + 128GB RAM):** Playwright, action log, orchestration
- **Ollama:** localhost:11434, Qwen3 14B with think: true
- No cloud dependencies. Fully local.

## What It Controls

### Tier 1 — Full DOM Access (Playwright)

All Electron apps expose Chrome DevTools Protocol. Launch with `--remote-debugging-port`:

| App | Launch Command (Windows) |
|-----|-------------------------|
| Chrome/Browser | Native Playwright support, no flag needed |
| VS Code | `code.exe --remote-debugging-port=9223` |
| Spotify | `Spotify.exe --remote-debugging-port=9227` |
| Slack | `slack.exe --remote-debugging-port=9222` |
| Discord | `Discord.exe --remote-debugging-port=9224` |
| Notion | `Notion.exe --remote-debugging-port=9226` |
| Any Electron app | `app.exe --remote-debugging-port=PORT` |

**Important:** App must be quit first, then relaunched with the flag.

### Tier 2 — Vision Fallback (existing)
- Native Win32 apps without CDP/DOM access
- Uses shadow desktop (WSL2 + Xvfb) + UI-TARS-72B

## Input Layer — Three Entry Points

All inputs feed into a single task queue:

1. **Voice** via NEO-TX — "Hey, go book that flight"
2. **Phone** via AlchemyCode — text command from iPhone
3. **Direct API** — POST /v1/agent/task

## Approval Gate

The agent runs autonomously UNTIL it reaches an irreversible action:
- Sending emails, deleting files, making purchases
- Agent does ALL prep work, pauses at the point of no return
- **PiP viewport** shows orange border = human decision needed
- Confirm via voice ("commit"), tap, or Enter
- Toggleable in OS settings app

## PiP Viewport (Picture-in-Picture)

- Minimizable corner window showing agent working step-by-step
- Like YouTube PiP on iPhone — minimize and do your own work
- Click to expand to full view (portal into the agent's browser)
- Orange border = needs human input
- Powered by CDP page streaming (or noVNC for Tier 2)

## Action Logging → NEO-RX

Every agent action is logged to the NEO-RX timeline:
- click @e7 on "Submit" button
- type @e5 "hello world"
- scroll down
- approval requested → approved/denied

Full searchable action history. Future: Nightwatch pattern analysis.

## The Accessibility Snapshot

What Playwright gives the model — structured, labeled, no pixels:

```
- heading "Good Evening" [ref=e1] [level=1]
- navigation "Main"
  - link "Home" [ref=e2]
  - link "Search" [ref=e3]
  - link "Your Library" [ref=e4]
- search "Search Spotify"
  - textbox "What do you want to play?" [ref=e5]
- list "Recently Played"
  - listitem "Daily Mix 1" [ref=e6]
  - listitem "Discover Weekly" [ref=e7]
- button "Play" [ref=e8]
- slider "Volume" [ref=e9] [value=75]
```

A 14B model reads this and maps "Play Discover Weekly" → `click @e7` then `click @e8`. Trivially easy.

## System Prompt

```
You are a GUI automation agent operating inside AlchemyOS.
You control applications through their accessibility tree.

RULES:
- Output exactly ONE action per turn
- Use the ref labels from the snapshot (e.g., @e5)
- If unsure, use "scroll down" to see more of the page
- If a page is loading, output "wait"
- If the task is complete, output "done"
- Never guess — if you can't find the element, say so

ACTION FORMAT (use exactly one per response):
  click @REF         — click an element
  type @REF "text"   — type text into an input
  scroll down        — scroll the page down
  scroll up          — scroll the page up
  key KEYNAME        — press a key (Enter, Tab, Escape, etc.)
  select @REF "option" — select from dropdown
  wait               — wait for page to load
  done               — task is complete

RESPONSE FORMAT:
Thought: [brief reasoning about what you see and what to do]
Action: [exactly one action from above]
```

## Model Configuration

- **Model:** qwen3:14b via Ollama
- **Think mode:** true (deep reasoning per step, speed irrelevant for background agent)
- **Temperature:** 0.1 (deterministic actions)
- **Max tokens:** 200 per action step
- **Timeout:** 30s per inference call
- **Endpoint:** http://localhost:11434/api/chat

## Repo Structure (New Modules)

```
alchemy/
  agent/
    core.py               — main loop: task → snapshot → LLM → action → repeat
    action_parser.py      — parse "click @e7" → Action dataclass (REPLACE existing)
    prompts.py            — system prompt + action format template
    llm_client.py         — Ollama Qwen3 14B client (think: true)
  playwright/
    browser.py            — launch headless Chromium, manage pages
    electron.py           — connect to Electron apps via CDP
    executor.py           — execute Action on page (click, type, scroll, key)
    snapshot.py           — capture + format accessibility tree for LLM context
  approval/
    gate.py               — detect irreversible actions, pause, signal viewport
  vision/                 — EXISTING UI-TARS (Tier 2, untouched)
  shadow/                 — EXISTING WSL2 + Xvfb (for Tier 2, untouched)
  router/                 — EXISTING context router (reused for task classification)
  models/                 — EXISTING Ollama client (extended for Qwen3)
  api/
    server.py             — task queue endpoint (voice/phone/direct all land here)
```

## Build Phases

### Phase 1: Browser Only (Day 1)
Get the loop working for headless Chromium on Windows.

1. Set up Playwright with Chromium (headless)
2. Write snapshot capture + formatter (accessibility tree → text)
3. Write LLM client for Qwen3 14B (think: true) via Ollama
4. Write action parser (parse LLM output → Action dataclass)
5. Write executor (Action → Playwright commands)
6. Write the main agent loop (snapshot → LLM → action → repeat)
7. Test: "Go to wikipedia and search for pole vault"

### Phase 2: Electron Apps (Day 2)
Extend to VS Code and Spotify.

1. Create launcher/connector for Electron apps via CDP
2. Connect Playwright via: `browser.connect_over_cdp("http://localhost:PORT")`
3. Test on Spotify: "Play Discover Weekly"
4. Test on VS Code: "Open the file explorer and create a new file called test.py"

### Phase 3: Approval Gate + PiP (Day 3)
Wire the safety layer and viewport.

1. Implement irreversible action detection
2. Build approval gate (pause → signal → wait for confirm)
3. Wire PiP viewport via CDP page streaming
4. Test: "Send an email" → agent preps everything → pauses → orange border → confirm

### Phase 4: Integration (Day 4+)
Wire into the full stack.

1. Connect task queue to NEO-TX voice input
2. Connect task queue to AlchemyCode phone input
3. Pipe action logs to NEO-RX timeline
4. OS settings app control panel

## Why This Works Better Than UI-TARS

| | UI-TARS Approach | Playwright Approach |
|---|---|---|
| **Input** | Screenshot (pixels) | Accessibility tree (structured text) |
| **Model needed** | 72B VLM | 14B text LLM (already running) |
| **Grounding** | Must find coordinates in image | Refs provided by Playwright |
| **Speed** | 15+ sec per action | 2-5 sec per action |
| **Reliability** | Misclicks on wrong pixels | Deterministic ref-based clicks |
| **GPU cost** | Both GPUs locked (47GB) | One GPU, shared with chat (9GB) |
| **Infrastructure** | WSL2 + Xvfb + noVNC | Just Playwright on Windows |
| **Coverage** | Any screen | Web + Electron (80-95% of daily use) |

## Key Reference Links

- Playwright Docs: https://playwright.dev/python/
- Playwright Electron: https://playwright.dev/docs/api/class-electron
- Playwright MCP Electron: https://github.com/robertn702/playwright-mcp-electron
- Vercel Agent Browser (ref-based): https://github.com/vercel-labs/agent-browser
- Browser Use (89% WebVoyager): https://github.com/browser-use/browser-use

## One-Line Summary

**Playwright snapshots the accessibility tree → Qwen3 14B reads it and picks an action → Playwright executes it. Loop. No vision, no cloud, no extra GPU. The 80% solution that actually works.**
