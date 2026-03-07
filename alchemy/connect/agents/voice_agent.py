"""VoiceAgent — voice pipeline control through the AlchemyConnect tunnel.

Bridges to VoiceSystem (alchemy/voice/) for start/stop/mode/status.
The voice pipeline itself handles STT → LLM → TTS internally.
This agent just gives the phone remote control over it.

Protocol:
    -> {agent: "voice", type: "status"}
    <- {agent: "voice", type: "status", payload: {running: true, mode: "conversation", ...}}

    -> {agent: "voice", type: "start"}
    <- {agent: "voice", type: "started"}

    -> {agent: "voice", type: "stop"}
    <- {agent: "voice", type: "stopped"}

    -> {agent: "voice", type: "mode", payload: {mode: "command"}}
    <- {agent: "voice", type: "mode_changed", payload: {mode: "command"}}

    -> {agent: "voice", type: "say", payload: {text: "Hello from phone"}}
    <- {agent: "voice", type: "spoken", payload: {text: "Hello from phone"}}
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from alchemy.connect.protocol import AlchemyMessage
from alchemy.connect.router import ConnectAgent

logger = logging.getLogger(__name__)


class VoiceAgent(ConnectAgent):
    """Remote control for AlchemyVoice pipeline."""

    def __init__(self, app_state: Any) -> None:
        self._app_state = app_state

    @property
    def agent_id(self) -> str:
        return "voice"

    def describe(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "description": "Voice pipeline control (start/stop/mode/status/say)",
            "available": self._voice_system is not None,
            "types": {
                "status": "Get voice pipeline status",
                "start": "Start voice pipeline",
                "stop": "Stop voice pipeline",
                "mode": "Change voice mode (conversation/command/dictation/muted)",
                "say": "Speak text through PC speakers via TTS",
            },
        }

    @property
    def _voice_system(self):
        return getattr(self._app_state, "voice_system", None)

    async def handle(
        self,
        msg: AlchemyMessage,
        device_id: str,
    ) -> AsyncIterator[AlchemyMessage]:
        voice = self._voice_system

        if msg.type == "status":
            if not voice:
                yield AlchemyMessage(
                    agent="voice", type="status",
                    payload={"running": False, "available": False,
                             "reason": "Voice system not enabled"},
                    ref=msg.id,
                )
                return

            status = voice.status()
            yield AlchemyMessage(
                agent="voice", type="status",
                payload={**status.to_dict(), "available": True},
                ref=msg.id,
            )
            return

        if msg.type == "start":
            if not voice:
                yield self._not_available(msg)
                return

            if voice.is_running:
                yield AlchemyMessage(
                    agent="voice", type="started",
                    payload={"status": "already_running", "mode": voice.mode.value},
                    ref=msg.id,
                )
                return

            try:
                await voice.start()
                yield AlchemyMessage(
                    agent="voice", type="started",
                    payload={"status": "started"},
                    ref=msg.id,
                )
            except Exception as e:
                logger.exception("VoiceAgent start error")
                yield AlchemyMessage(
                    agent="voice", type="error",
                    payload={"reason": str(e)},
                    ref=msg.id,
                )
            return

        if msg.type == "stop":
            if not voice:
                yield self._not_available(msg)
                return

            if not voice.is_running:
                yield AlchemyMessage(
                    agent="voice", type="stopped",
                    payload={"status": "already_stopped"},
                    ref=msg.id,
                )
                return

            try:
                await voice.stop()
                yield AlchemyMessage(
                    agent="voice", type="stopped",
                    payload={"status": "stopped"},
                    ref=msg.id,
                )
            except Exception as e:
                logger.exception("VoiceAgent stop error")
                yield AlchemyMessage(
                    agent="voice", type="error",
                    payload={"reason": str(e)},
                    ref=msg.id,
                )
            return

        if msg.type == "mode":
            if not voice:
                yield self._not_available(msg)
                return

            mode_str = msg.payload.get("mode", "")
            try:
                from alchemy.voice import VoiceMode
                mode = VoiceMode(mode_str)
            except ValueError:
                yield AlchemyMessage(
                    agent="voice", type="error",
                    payload={
                        "reason": f"Invalid mode: {mode_str}",
                        "valid": ["conversation", "command", "dictation", "muted"],
                    },
                    ref=msg.id,
                )
                return

            voice.set_mode(mode)
            yield AlchemyMessage(
                agent="voice", type="mode_changed",
                payload={"mode": mode.value},
                ref=msg.id,
            )
            return

        if msg.type == "say":
            if not voice:
                yield self._not_available(msg)
                return

            text = msg.payload.get("text", "").strip()
            if not text:
                yield AlchemyMessage(
                    agent="voice", type="error",
                    payload={"reason": "Empty text"},
                    ref=msg.id,
                )
                return

            try:
                # Use the TTS engine to speak on PC speakers
                if hasattr(voice, "_pipeline") and voice._pipeline:
                    tts = getattr(voice._pipeline, "_tts", None)
                    audio_out = getattr(voice._pipeline, "_audio_output", None)
                    if tts and audio_out:
                        await tts.speak(text, audio_out)
                        yield AlchemyMessage(
                            agent="voice", type="spoken",
                            payload={"text": text},
                            ref=msg.id,
                        )
                        return

                # Fallback: voice system doesn't have TTS accessible
                yield AlchemyMessage(
                    agent="voice", type="error",
                    payload={"reason": "TTS not accessible (pipeline not running)"},
                    ref=msg.id,
                )
            except Exception as e:
                logger.exception("VoiceAgent say error")
                yield AlchemyMessage(
                    agent="voice", type="error",
                    payload={"reason": str(e)},
                    ref=msg.id,
                )
            return

        yield AlchemyMessage(
            agent="voice", type="error",
            payload={"reason": f"Unknown type: {msg.type}"},
            ref=msg.id,
        )

    def _not_available(self, msg: AlchemyMessage) -> AlchemyMessage:
        return AlchemyMessage(
            agent="voice", type="error",
            payload={"reason": "Voice system not available"},
            ref=msg.id,
        )
