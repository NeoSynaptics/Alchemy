# Alchemy Codebase Review

**Date:** 2026-03-08
**Scope:** Full codebase review — architecture, security, correctness, code quality, convention compliance

---

## Executive Summary

Alchemy is a well-architected local-first AI engine with strong modular isolation enforced by import-linter contracts. The manifest/registry/contract system is a solid foundation for plugin-style extensibility. The APU orchestrator's tier-based eviction and dual-GPU placement logic is thoughtfully designed.

However, several critical issues need attention — most urgently around **security** (no auth enforcement, dangerous CORS), **configuration drift** (nested vs flat settings divergence), and **version inconsistency**. Below are findings organized by severity.

---

## Critical Issues

### 1. No Authentication Enforcement (Security)

**Files:** `alchemy/server.py`, `alchemy/security/__init__.py`, `config/settings.py`

Despite `AuthSettings` existing with `require` and `token` fields, **no auth middleware is applied anywhere**. The `alchemy/security/` module is an empty stub (only a docstring). Every endpoint — including desktop automation, browser control, and voice pipeline — is publicly accessible to anyone who can reach port 8000.

Given that this system can:
- Execute browser actions via Playwright
- Send native Windows input via `DesktopController`
- Control voice pipeline and model loading

This is a **serious exposure** if the server is reachable beyond localhost.

**Recommendation:** Implement a FastAPI dependency that reads `settings.auth` and enforces bearer token validation. Apply it globally or per-router.

### 2. Dangerous CORS Configuration (Security)

**File:** `alchemy/server.py:351-357`

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,  # <-- dangerous with wildcard origins
    allow_methods=["*"],
    allow_headers=["*"],
)
```

`allow_origins=["*"]` combined with `allow_credentials=True` is a well-known misconfiguration. While modern browsers block this combination, it signals intent to accept credentials from any origin. With auth disabled, the risk is compounded.

**Recommendation:** Either restrict `allow_origins` to known hosts (e.g., `http://localhost:5173` for the UI) or set `allow_credentials=False`.

### 3. Three Different Version Strings

| Location | Version |
|----------|---------|
| `pyproject.toml` | `0.4.0` |
| `alchemy/__init__.py:3` | `0.1.0` |
| `alchemy/server.py:347` (FastAPI metadata) | `0.2.0` |
| `alchemy/server.py:406` (`/health` response) | `0.4.0` |

**Recommendation:** Single source of truth. Read `__version__` from `pyproject.toml` (via `importlib.metadata`) and reference it everywhere.

### 4. Nested vs Flat Settings Divergence (Data Integrity)

**File:** `config/settings.py`

Every setting is duplicated: once in a nested group (e.g., `OllamaSettings.host`) and once as a flat field (`Settings.ollama_host`). These are **completely independent** — they read from different env vars (`OLLAMA__HOST` vs `OLLAMA_HOST`) and have no sync mechanism.

Code using `settings.ollama_host` and code using `settings.ollama.host` can silently see different values if the environment only sets one form.

`server.py` exclusively uses flat fields, making the nested groups effectively dead code at runtime.

**Recommendation:** Either wire `server.py` to use nested settings, or create `@property` aliases that delegate flat fields to their nested counterpart. Don't maintain two independent copies.

---

## High-Priority Issues

### 5. Unconditional Voice API Imports Will Crash When Voice Is Disabled

**File:** `alchemy/server.py:381-383`

```python
from alchemy.voice.api import callbacks as voice_callbacks
from alchemy.voice.api import chat as voice_chat
from alchemy.voice.api import voice as voice_control
```

These are top-level imports that execute regardless of `settings.voice_enabled`. If the `voice` optional dependencies (faster-whisper, openwakeword, etc.) are not installed, **the entire server fails to start**, even when voice is disabled.

**Recommendation:** Guard with `if settings.voice_enabled:` or wrap in `try/except ImportError`.

### 6. Tier String Casing Mismatch in Contract Validation (Likely Bug)

**File:** `alchemy/contracts.py:72, 101-102`

```python
_TIER_RANK = {"resident": 0, "user_active": 1, "agent": 2, "warm": 3, "cold": 4}
# ...
result.tier_ok = _tier_meets_minimum(card.current_tier.value, req.min_tier)
```

`ModelTier` enum values are lowercase (`"resident"`, `"warm"`, etc.), so this works. However, the lookup relies on exact string matching with no normalization. If anyone passes a tier string like `"Warm"` or `"WARM"`, it falls through to the default of 99. Consider using `.lower()` in `_tier_meets_minimum` for robustness.

### 7. `ModelRegistry` Claims Thread-Safety But Isn't

**File:** `alchemy/apu/registry.py:92`

The docstring says "Thread-safe model registry" but uses a plain `dict` with no locking. In an async context with potential concurrent model loads/evictions, `_models` could see race conditions. The `StackOrchestrator` also has no locking around multi-step operations (check free VRAM → evict → load).

**Recommendation:** Add an `asyncio.Lock` around critical sections in `StackOrchestrator.load_model`, `_make_room`, and `rebalance`.

### 8. Private Attribute Access Across Module Boundaries

**File:** `alchemy/server.py:289`
```python
connect.register_agent(ImageAgent(app.state, gpu_guard=connect._hub._gpu_semaphore))
```

Two layers of private attribute access. This is fragile and violates encapsulation.

**File:** `alchemy/server.py:398`
```python
desktop_mode = desktop_agent._controller.mode if desktop_agent else None
```

Accessing private `_controller` on a public, unauthenticated `/health` endpoint.

**Recommendation:** Expose these via public properties on their respective classes.

---

## Medium-Priority Issues

### 9. Monolithic Lifespan Function (~320 lines)

**File:** `alchemy/server.py:38-342`

The `lifespan()` function initializes 10+ subsystems in sequence. Each block follows the same pattern (check enabled → import → construct → start → assign to app.state). This should be broken into helper functions for readability and testability.

### 10. Empty Auth Token as Default

**File:** `config/settings.py:40-41`

```python
token: str = ""
require: bool = False
```

If someone sets `require_auth=True` without setting a token, the empty string becomes the valid authentication credential. The system should refuse to start (or warn loudly) if auth is required but no token is configured.

### 11. No Settings Validation

**File:** `config/settings.py`

No `@field_validator` decorators on any field. Values like `voice_vad_aggressiveness` (WebRTC VAD valid range: 0-3), `screenshot_jpeg_quality` (valid: 1-100), temperatures (valid: 0.0-2.0), and port numbers are unconstrained. Invalid values will produce cryptic runtime errors instead of clear startup failures.

### 12. `optional_missing` Doesn't Check Tier

**File:** `alchemy/contracts.py:62-68`

```python
@property
def optional_missing(self) -> list[str]:
    return [
        r.requirement.capability
        for r in self.results
        if not r.requirement.required and not r.available
    ]
```

This only checks `not r.available` but ignores `r.tier_ok`. An optional model that is available but fails the tier check won't appear in either `missing` or `optional_missing`.

### 13. Module Registry Cache Has No Invalidation

**File:** `alchemy/registry.py:22-23`

```python
if _REGISTRY:
    return _REGISTRY
```

Once populated, `discover()` never re-scans. The `reset()` function exists but is documented as "for testing only." If modules are added dynamically, they won't be discovered.

### 14. Python Version Requirement Mismatch

**File:** `pyproject.toml:9`

`requires-python = ">=3.12"` but the CI/dev environment has Python 3.11 installed. Tests cannot run without the correct Python version.

### 15. `host: str = "0.0.0.0"` Default

**File:** `config/settings.py:33, 267`

Binding to all interfaces by default is convenient for development but risky in production. Combined with the lack of auth, this means any device on the network can control the system.

---

## Architecture Observations (Positive)

### Import Boundary Contracts
The `.importlinter` configuration is comprehensive with 11 contracts enforcing:
- Core/infra never import from features
- Features never import laterally
- Registry/manifest are fully independent
- GPU orchestrator is isolated from all features

This is a best-practice approach to modular architecture. The contracts match what CLAUDE.md documents.

### APU Design
The tier-based eviction system (`RESIDENT → USER_ACTIVE → AGENT → WARM → COLD`) with "no model is immune" semantics and RAM-first eviction is well-thought-out. The eviction ordering (app → infra → core, then LRU) is a pragmatic approach.

### Voice Public Interface
`VoiceSystem` properly hides all internals (Whisper, Fish Speech, VRAM management) behind a clean interface. The `_build_pipeline()` pattern with lazy imports is correct for optional dependencies.

### Module Manifest System
The `ModuleManifest` + `ModelRequirement` + `ContractReport` chain is a clean declarative approach. Apps declare what they need, core validates and resolves — good separation of concerns.

---

## Import Linter Gaps

The `.importlinter` contracts have a few gaps relative to CLAUDE.md conventions:

1. **`core-independence` doesn't forbid**: `alchemy.click`, `alchemy.gate`, `alchemy.desktop`, `alchemy.voice`, `alchemy.word`, `alchemy.playwright`. CLAUDE.md says "Core and adapters NEVER import from features."
2. **Missing contracts for**: `alchemy.connect`, `alchemy.agents`, `alchemy.clients` — these modules exist but have no import boundary enforcement.
3. **`adapters-independence` doesn't forbid**: `alchemy.click`, `alchemy.gate`, `alchemy.desktop`, `alchemy.voice`, `alchemy.word`, `alchemy.playwright`.

---

## Test Suite

- 99 test files with good structural mirroring of the module layout
- Custom markers (`live`, `integration`, `gpu`) for separating test tiers
- Default test run excludes heavy tests: `-m 'not integration and not live and not gpu'`
- Uses `pytest-asyncio` with `asyncio_mode = "auto"`
- Cannot currently run due to Python 3.11 vs >=3.12 requirement mismatch

---

## Summary of Recommendations (Priority Order)

| # | Severity | Issue | Fix |
|---|----------|-------|-----|
| 1 | **Critical** | No auth enforcement | Implement auth middleware using existing `AuthSettings` |
| 2 | **Critical** | CORS wildcard + credentials | Restrict origins or disable credentials |
| 3 | **Critical** | Settings divergence | Wire server.py to nested settings or create property aliases |
| 4 | **High** | Voice imports crash without deps | Guard with conditional import |
| 5 | **High** | Three version strings | Single source via `importlib.metadata` |
| 6 | **High** | No async locking in APU | Add `asyncio.Lock` to orchestrator |
| 7 | **Medium** | Private attribute access | Expose via public properties |
| 8 | **Medium** | Monolithic lifespan | Extract init helpers |
| 9 | **Medium** | No settings validation | Add `@field_validator` for bounded values |
| 10 | **Medium** | Import linter gaps | Extend contracts for all feature modules |
| 11 | **Low** | Empty auth token default | Validate at startup |
| 12 | **Low** | Python version mismatch | Ensure CI uses Python 3.12+ |
