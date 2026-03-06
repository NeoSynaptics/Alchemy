"""Tray manager — launches PyQt6 in a daemon thread.

Owns the GUI lifecycle. Created in server lifespan, starts a daemon thread
that runs QApplication.exec(). Stops cleanly on server shutdown.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.settings import Settings
    from alchemy.voice.tray.events import TrayEventBus

logger = logging.getLogger(__name__)


class TrayManager:
    """Manages the PyQt6 system tray in a background daemon thread."""

    def __init__(self, event_bus: TrayEventBus, settings: Settings) -> None:
        self._event_bus = event_bus
        self._settings = settings
        self._thread: threading.Thread | None = None
        self._qt_app = None  # QApplication reference (set in thread)

    def start(self) -> None:
        """Launch the tray GUI in a daemon thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("Tray already running")
            return

        self._thread = threading.Thread(
            target=self._run_gui,
            name="neo-tray",
            daemon=True,
        )
        self._thread.start()
        logger.info("Tray thread started")

    def stop(self) -> None:
        """Signal Qt to quit and wait for thread to finish."""
        if self._qt_app is not None:
            try:
                from PyQt6.QtCore import QMetaObject, Qt, Q_ARG

                QMetaObject.invokeMethod(
                    self._qt_app, "quit", Qt.ConnectionType.QueuedConnection
                )
            except Exception:
                logger.debug("Qt quit signal failed (already stopped?)")

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            logger.info("Tray thread stopped")

        self._thread = None
        self._qt_app = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run_gui(self) -> None:
        """Thread target: create QApplication, tray icon, and run Qt event loop."""
        try:
            from PyQt6.QtWidgets import QApplication

            from alchemy.voice.tray.icon import NeoTrayIcon

            app = QApplication([])
            app.setQuitOnLastWindowClosed(False)
            self._qt_app = app

            tray_icon = NeoTrayIcon(
                event_bus=self._event_bus,
                settings=self._settings,
                app=app,
            )
            tray_icon.show()

            app.exec()
        except ImportError:
            logger.warning("PyQt6 not installed — tray disabled. Run: pip install -e '.[tray]'")
        except Exception:
            logger.exception("Tray thread crashed")
        finally:
            self._qt_app = None
