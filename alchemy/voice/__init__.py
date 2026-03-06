"""AlchemyVoice — first-class voice subsystem of Alchemy core.

Public interface:
    VoiceSystem  — start/stop/configure voice (hides model internals)
    VoiceMode    — user-facing behavior modes (conversation, command, dictation, muted)
    VoiceStatus  — serializable status snapshot for GUI/API
    MANIFEST     — module contract for GPU model requirements

Everything else (pipeline, STT, TTS, VRAM, models, router) is internal.
External code should only import from this module.
"""

from alchemy.voice.interface import VoiceMode, VoiceStatus, VoiceSystem
from alchemy.voice.manifest import MANIFEST

__all__ = ["MANIFEST", "VoiceSystem", "VoiceMode", "VoiceStatus"]
