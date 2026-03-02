"""Voice pipeline — Whisper STT + Piper TTS + wake word detection.

Voice is a general input/output layer that lives in Alchemy (not NEO-TX).
Mic → Whisper STT → 14B interprets → routes to NEO-TX if GUI needed,
or answers directly if it doesn't. NEO-TX never touches audio.
"""
