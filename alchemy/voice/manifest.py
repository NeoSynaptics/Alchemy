"""Voice manifest -- STT + TTS pipeline (core, always resident)."""

from alchemy.manifest import ModelRequirement, ModuleManifest

MANIFEST = ModuleManifest(
    id="voice",
    name="Voice Pipeline",
    description="Wake word detection, speech-to-text, and text-to-speech.",
    settings_prefix="voice_",
    enabled_key="voice_enabled",
    requires=["adapters"],
    tier="core",
    api_prefix="/v1",
    api_tags=["voice"],
    models=[
        ModelRequirement(
            capability="stt",
            required=True,
            preferred_model="whisper-large-v3",
            min_tier="resident",
        ),
        ModelRequirement(
            capability="tts",
            required=True,
            preferred_model="fish-speech-s1",
            min_tier="resident",
        ),
    ],
)
