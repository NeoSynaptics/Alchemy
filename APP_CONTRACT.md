# Alchemy App Contract — Developer Guide

> **Rule:** No app can use Alchemy Core without declaring a contract. This is enforced at the API level — endpoints return HTTP 503 if the contract is not satisfied.

## What Is a Contract?

Every Alchemy module that needs LLM models must declare **what it needs** in its manifest. The core validates these declarations against the GPU/RAM fleet on startup. If required models are missing, the module's API endpoints are blocked until the models are pulled.

**Apps declare needs. Core provides models. Never the other way around.**

## Quick Start — New Module Checklist

```
1. alchemy/<name>/manifest.py     ← REQUIRED (contract lives here)
2. alchemy/<name>/__init__.py     ← Public API + __all__
3. alchemy/api/<name>_api.py      ← API routes with contract guard
4. config/settings.py             ← Nested BaseModel settings group
5. .importlinter                  ← Import boundary contract
6. tests/test_<name>.py           ← Tests
```

## Step 1: Write the Manifest

Create `alchemy/<name>/manifest.py`:

```python
"""My module manifest."""

from alchemy.manifest import ModelRequirement, ModuleManifest

MANIFEST = ModuleManifest(
    id="mymod",                          # Unique ID (lowercase, no spaces)
    name="My Module",                    # Human-readable name
    description="One sentence explaining what it does.",

    # Settings
    settings_prefix="mymod_",            # Maps to config/settings.py group
    enabled_key="mymod_enabled",         # On/off toggle

    # Dependencies
    requires=["adapters"],               # Other modules this depends on
    env_keys=["MY_API_KEY"],             # Required environment variables

    # Classification
    tier="app",                          # "core" | "infra" | "app"

    # API
    api_prefix="/v1",                    # Route prefix
    api_tags=["mymod"],                  # OpenAPI tags

    # MODEL CONTRACT — the critical part
    models=[
        ModelRequirement(
            capability="reasoning",       # What the model must do
            required=True,                # True = app won't work without it
            preferred_model="qwen3:14b",  # Hint — core may substitute
            min_tier="warm",              # "resident" | "warm" | "cold"
        ),
        ModelRequirement(
            capability="vision",
            required=False,               # Optional — app degrades gracefully
            preferred_model="qwen2.5vl:7b",
            min_tier="warm",
            context_tokens=2048,          # VRAM budgeting hint
        ),
    ],
)
```

## Step 2: Wire the Contract Guard

In your API file `alchemy/api/<name>_api.py`:

```python
from fastapi import APIRouter, Depends
from alchemy.api.contract_guard import require_contract

# This guard blocks ALL endpoints in this router if the contract is unsatisfied
router = APIRouter(
    tags=["mymod"],
    dependencies=[Depends(require_contract("mymod"))],
)

@router.post("/mymod/task")
async def submit_task(...):
    ...
```

**What happens when the contract fails:**

```json
HTTP 503
{
    "detail": {
        "error": "model_contract_unsatisfied",
        "module": "mymod",
        "missing_capabilities": ["reasoning"],
        "message": "Module 'mymod' requires models with capabilities: ['reasoning']. Pull the required models first."
    }
}
```

## Step 3: Add Settings Group

In `config/settings.py`, add a nested BaseModel:

```python
class MyModSettings(BaseModel):
    enabled: bool = False
    model: str = "qwen3:14b"
    max_steps: int = 20

class Settings(BaseSettings):
    mymod: MyModSettings = MyModSettings()
    mymod_enabled: bool = False  # Flat compat toggle
```

## Step 4: Add Import Boundary

In `.importlinter`:

```ini
[importlinter:contract:mymod-isolation]
name = mymod does not import from other features
type = forbidden
source_modules = alchemy.mymod
forbidden_modules = alchemy.gate, alchemy.research, alchemy.desktop
```

## ModelRequirement Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `capability` | str | — | What the model does: "reasoning", "vision", "coding", "embedding", "classification", "voice" |
| `required` | bool | `True` | `True` = endpoint blocked without it. `False` = nice-to-have. |
| `preferred_model` | str? | `None` | Hint: "qwen3:14b". Core may use any model with matching capability. |
| `min_tier` | str | `"warm"` | Minimum fleet tier: "resident" (always loaded) > "warm" (in RAM) > "cold" (on disk) |
| `context_tokens` | int? | `None` | Max context the app needs. Used for VRAM budgeting. |

## ModuleManifest Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | str | — | Unique module identifier |
| `name` | str | — | Human-readable display name |
| `description` | str | — | One-sentence description |
| `settings_prefix` | str | `""` | Maps to nested settings group |
| `enabled_key` | str? | `None` | On/off toggle field name |
| `requires` | list[str] | `[]` | Module dependencies |
| `env_keys` | list[str] | `[]` | Required environment variables |
| `tier` | str | `"app"` | "core" (locked), "infra" (plumbing), "app" (features) |
| `api_prefix` | str? | `None` | Route prefix for API endpoints |
| `api_tags` | list[str] | `[]` | OpenAPI tags |
| `models` | list[ModelRequirement] | `[]` | **The model contract** |

## Discovery API

Check any module's contract status at runtime:

```bash
# All modules with contract status
GET http://localhost:8000/v1/modules

# Single module with detailed contract breakdown
GET http://localhost:8000/v1/modules/gate
```

Response includes:
- `contract_satisfied` — bool, all required models available
- `contract_missing` — list of missing required capabilities
- `contract_optional_missing` — list of missing optional capabilities
- `contract_details` — per-requirement breakdown with matched model names

## Enforcement Points

| When | What | Effect |
|------|------|--------|
| **Startup** | `validate_contracts()` runs in server lifespan | Logs warnings for missing models |
| **API call** | `require_contract()` dependency on router | HTTP 503 if required models missing |
| **Discovery** | `GET /v1/modules` | Shows contract status for setup wizards |

## Existing Module Contracts

| Module | ID | Required | Optional |
|--------|----|----------|----------|
| Agent Kernel | `core` | reasoning | vision |
| Desktop Agent | `desktop` | vision | — |
| Gate Reviewer | `gate` | reasoning | — |
| AlchemyBrowser | `research` | reasoning | embedding |
| AlchemyClick | `click` | reasoning | vision+clicking |
| Context Router | `router` | — | classification |
| Cloud AI Bridge | `cloud` | — | — (env_keys only) |
| Shadow Desktop | `shadow` | — | — |
| GPU Orchestrator | `gpu` | — | — |
| LLM Adapters | `adapters` | — | — |

## Rules That Cannot Be Broken

1. **Apps NEVER load models directly.** Declare in manifest, core provides.
2. **Every module with API routes MUST have a manifest.** No manifest = not discoverable = doesn't exist.
3. **API routes MUST use `require_contract()`.** No guard = contract isn't enforced = defeats the purpose.
4. **Feature modules NEVER import each other.** Lateral isolation is absolute.
5. **No new flat settings fields.** Use nested `BaseModel` groups only.
6. **Secrets go in `env_keys`, not hardcoded.** Setup wizard reads these.
