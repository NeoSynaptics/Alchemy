"""Thread-safe event bus for FastAPI ↔ PyQt6 communication.

No GUI dependency — pure Python. The FastAPI thread pushes events,
the GUI thread polls them via QTimer. Approval responses flow back
through asyncio.Future set from the GUI thread.
"""

from __future__ import annotations

import asyncio
import logging
import queue
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TrayEvent(str, Enum):
    APPROVAL_REQUEST = "approval_request"
    NOTIFICATION = "notification"
    TASK_UPDATE = "task_update"
    VOICE_STATE = "voice_state"


@dataclass
class TrayMessage:
    event: TrayEvent
    data: dict[str, Any]
    response_future: asyncio.Future | None = field(default=None, repr=False)


class TrayEventBus:
    """Thread-safe bridge between FastAPI (async) and PyQt6 (Qt event loop).

    - FastAPI calls ``push()`` to send events to the GUI thread.
    - GUI thread calls ``poll()`` from a QTimer to receive events.
    - GUI thread calls ``respond()`` to resolve an approval Future.
    """

    def __init__(self) -> None:
        self._to_gui: queue.Queue[TrayMessage] = queue.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the asyncio event loop (call from FastAPI thread at startup)."""
        self._loop = loop

    def push(self, msg: TrayMessage) -> None:
        """Send an event to the GUI thread (thread-safe)."""
        self._to_gui.put_nowait(msg)
        logger.debug("Event pushed: %s", msg.event.value)

    def poll(self) -> TrayMessage | None:
        """Non-blocking poll from GUI thread. Returns None if empty."""
        try:
            return self._to_gui.get_nowait()
        except queue.Empty:
            return None

    def pending_count(self) -> int:
        """Number of events waiting in the queue."""
        return self._to_gui.qsize()

    def respond(self, msg: TrayMessage, approved: bool, reason: str = "") -> None:
        """Resolve an approval Future from the GUI thread (thread-safe).

        Uses ``call_soon_threadsafe`` to set the result on the asyncio loop.
        """
        future = msg.response_future
        if future is None:
            logger.warning("respond() called on message with no Future")
            return

        if self._loop is None:
            logger.error("No asyncio loop bound — cannot resolve Future")
            return

        def _set_result():
            if not future.done():
                future.set_result(approved)

        self._loop.call_soon_threadsafe(_set_result)
        logger.debug("Approval response: approved=%s reason=%s", approved, reason)
