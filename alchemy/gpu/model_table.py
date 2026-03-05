"""Internal model priority table — maps capability tags to ranked model preferences.

CONFIDENTIAL: This table defines Alchemy's model selection strategy.
When an app declares capability tags, the resolver uses this table to pick
the best model from the fleet.

How it works:
1. App declares tags: ["vision", "text"] in ModelRequirement
2. Resolver scores each fleet model against those tags using this table
3. Best match wins. If developer pinned a preferred_model, that takes priority.

Tag vocabulary (growing list):
    vision      — needs to see screenshots/images
    text        — text generation, completion, chat
    reasoning   — multi-step thinking, planning, chain-of-thought
    coding      — code generation, completion, refactoring
    embedding   — vector embeddings for search/RAG
    voice       — speech (STT or TTS)
    stt         — speech-to-text specifically
    tts         — text-to-speech specifically
    agent       — autonomous task execution
    gate        — tool call review / approval
    clicking    — GUI coordinate prediction
    escalation  — vision fallback for stuck agents
    desktop     — native Windows automation
    classification — intent routing
    completion  — fast code/text completion
    distillation — knowledge distillation / deep reasoning
    conversation — casual chat
    routing     — request classification

Combo logic:
    ["vision", "text"]      → VLM (vision-language model)
    ["vision", "clicking"]  → GUI grounding model
    ["text", "reasoning"]   → large reasoning model
    ["text", "coding"]      → code-specialized model
    ["voice", "stt"]        → speech-to-text model
    ["voice", "tts"]        → text-to-speech model

Single tag falls back to the best model for that capability.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelCandidate:
    """A model name with a priority score for a capability match."""

    name: str
    score: int  # Lower = better (1 = top pick)


# ─── THE TABLE ───────────────────────────────────────────────────────────
# Key: frozenset of capability tags
# Value: ordered list of model candidates (best first)
#
# When multiple tags match, the most specific combo wins.
# If no combo matches, individual tags are scored and summed.

COMBO_TABLE: dict[frozenset[str], list[ModelCandidate]] = {
    # === Vision + Language ===
    frozenset({"vision", "text"}): [
        ModelCandidate("qwen2.5vl:7b", 1),
    ],
    frozenset({"vision", "clicking"}): [
        ModelCandidate("gui-actor-7b", 1),
        ModelCandidate("qwen2.5vl:7b", 2),
    ],
    frozenset({"vision", "escalation"}): [
        ModelCandidate("qwen2.5vl:7b", 1),
    ],
    frozenset({"vision", "desktop"}): [
        ModelCandidate("qwen2.5vl:7b", 1),
    ],

    # === Reasoning ===
    frozenset({"text", "reasoning"}): [
        ModelCandidate("qwen3:14b", 1),
        ModelCandidate("deepseek-r1:14b", 2),
    ],
    frozenset({"reasoning", "agent"}): [
        ModelCandidate("qwen3:14b", 1),
        ModelCandidate("deepseek-r1:14b", 2),
    ],
    frozenset({"reasoning", "gate"}): [
        ModelCandidate("qwen3:14b", 1),
    ],
    frozenset({"reasoning", "distillation"}): [
        ModelCandidate("deepseek-r1:14b", 1),
        ModelCandidate("qwen3:14b", 2),
    ],

    # === Coding ===
    frozenset({"text", "coding"}): [
        ModelCandidate("qwen2.5-coder:14b", 1),
        ModelCandidate("starcoder2:3b", 2),
        ModelCandidate("qwen3:14b", 3),
    ],
    frozenset({"coding", "completion"}): [
        ModelCandidate("starcoder2:3b", 1),
        ModelCandidate("qwen2.5-coder:14b", 2),
    ],

    # === Voice ===
    frozenset({"voice", "stt"}): [
        ModelCandidate("whisper-large-v3", 1),
    ],
    frozenset({"voice", "tts"}): [
        ModelCandidate("fish-speech-s1", 1),
    ],

    # === Conversation ===
    frozenset({"text", "conversation"}): [
        ModelCandidate("qwen3:14b", 1),
        ModelCandidate("qwen3:3b", 2),
    ],

    # === Classification / Routing ===
    frozenset({"classification", "routing"}): [
        ModelCandidate("deberta-v3-router", 1),
    ],
}

# ─── SINGLE-TAG FALLBACK TABLE ──────────────────────────────────────────
# Used when no combo matches. Each tag maps to its ranked model list.

SINGLE_TAG_TABLE: dict[str, list[ModelCandidate]] = {
    "vision": [
        ModelCandidate("qwen2.5vl:7b", 1),
        ModelCandidate("gui-actor-7b", 2),
    ],
    "text": [
        ModelCandidate("qwen3:14b", 1),
        ModelCandidate("qwen3:3b", 2),
    ],
    "reasoning": [
        ModelCandidate("qwen3:14b", 1),
        ModelCandidate("deepseek-r1:14b", 2),
    ],
    "coding": [
        ModelCandidate("qwen2.5-coder:14b", 1),
        ModelCandidate("starcoder2:3b", 2),
        ModelCandidate("qwen3:14b", 3),
    ],
    "embedding": [
        ModelCandidate("nomic-embed-text", 1),
    ],
    "voice": [
        ModelCandidate("whisper-large-v3", 1),
        ModelCandidate("fish-speech-s1", 2),
    ],
    "stt": [
        ModelCandidate("whisper-large-v3", 1),
    ],
    "tts": [
        ModelCandidate("fish-speech-s1", 1),
    ],
    "agent": [
        ModelCandidate("qwen3:14b", 1),
    ],
    "gate": [
        ModelCandidate("qwen3:14b", 1),
    ],
    "clicking": [
        ModelCandidate("gui-actor-7b", 1),
        ModelCandidate("qwen2.5vl:7b", 2),
    ],
    "escalation": [
        ModelCandidate("qwen2.5vl:7b", 1),
    ],
    "desktop": [
        ModelCandidate("qwen2.5vl:7b", 1),
    ],
    "classification": [
        ModelCandidate("deberta-v3-router", 1),
    ],
    "routing": [
        ModelCandidate("deberta-v3-router", 1),
    ],
    "completion": [
        ModelCandidate("starcoder2:3b", 1),
        ModelCandidate("qwen2.5-coder:14b", 2),
    ],
    "distillation": [
        ModelCandidate("deepseek-r1:14b", 1),
    ],
    "conversation": [
        ModelCandidate("qwen3:14b", 1),
        ModelCandidate("qwen3:3b", 2),
    ],
}

# ─── ALL KNOWN TAGS ─────────────────────────────────────────────────────
# Canonical vocabulary. Validated at manifest load time.

ALL_TAGS: frozenset[str] = frozenset(SINGLE_TAG_TABLE.keys())
