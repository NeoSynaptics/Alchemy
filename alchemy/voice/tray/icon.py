"""System tray icon — QSystemTrayIcon with context menu and event handler.

Runs in the GUI thread. Polls TrayEventBus via QTimer every 100ms
to receive events from the FastAPI thread.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

if TYPE_CHECKING:
    from config.settings import Settings
    from alchemy.voice.tray.events import TrayEventBus

logger = logging.getLogger(__name__)

_POLL_INTERVAL_MS = 100


class NeoTrayIcon(QSystemTrayIcon):
    """System tray icon with context menu and event polling."""

    def __init__(
        self,
        event_bus: TrayEventBus,
        settings: Settings,
        app: QApplication,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._event_bus = event_bus
        self._settings = settings
        self._app = app
        self._viewport_window = None

        self._setup_icon()
        self._setup_menu()
        self._setup_timer()

    def _setup_icon(self) -> None:
        """Set the tray icon. Uses a built-in icon as placeholder."""
        icon = QIcon.fromTheme("computer")
        if icon.isNull():
            # Fallback: use the application icon
            icon = self._app.style().standardIcon(
                self._app.style().StandardPixmap.SP_ComputerIcon
            )
        self.setIcon(icon)
        self.setToolTip("Neo — Smart AI Interface")

    def _setup_menu(self) -> None:
        """Build the right-click context menu."""
        menu = QMenu()

        self._voice_action = QAction("Voice: Start", menu)
        self._voice_action.triggered.connect(self._toggle_voice)
        menu.addAction(self._voice_action)

        menu.addSeparator()

        viewport_action = QAction("Open Viewport", menu)
        viewport_action.triggered.connect(self._open_viewport)
        menu.addAction(viewport_action)

        menu.addSeparator()

        quit_action = QAction("Quit", menu)
        quit_action.triggered.connect(self._quit)
        menu.addAction(quit_action)

        self.setContextMenu(menu)

    def _setup_timer(self) -> None:
        """Start polling the event bus every 100ms."""
        self._timer = QTimer()
        self._timer.timeout.connect(self._process_events)
        self._timer.start(_POLL_INTERVAL_MS)

    def _process_events(self) -> None:
        """Poll event bus and handle any pending messages."""
        from alchemy.voice.tray.events import TrayEvent

        while True:
            msg = self._event_bus.poll()
            if msg is None:
                break

            if msg.event == TrayEvent.APPROVAL_REQUEST:
                self._handle_approval(msg)
            elif msg.event == TrayEvent.NOTIFICATION:
                self._handle_notification(msg)
            elif msg.event == TrayEvent.TASK_UPDATE:
                self._handle_task_update(msg)
            elif msg.event == TrayEvent.VOICE_STATE:
                self._handle_voice_state(msg)

    def _handle_approval(self, msg) -> None:
        """Show approval dialog for APPROVE-tier action."""
        from alchemy.voice.tray.dialogs import ApprovalDialog

        dialog = ApprovalDialog(msg=msg, event_bus=self._event_bus)
        dialog.exec()

    def _handle_notification(self, msg) -> None:
        """Show toast notification for NOTIFY-tier action."""
        data = msg.data
        title = f"Neo — {data.get('action', {}).get('action', 'Action')}"
        body = data.get("message", "Task update")
        self.showMessage(title, body, QSystemTrayIcon.MessageIcon.Information, 5000)

    def _handle_task_update(self, msg) -> None:
        """Update tooltip with task status."""
        data = msg.data
        status = data.get("status", "unknown")
        step = data.get("current_step", 0)
        message = data.get("message", "")
        tooltip = f"Neo — Task: {status} (step {step})"
        if message:
            tooltip += f"\n{message}"
        self.setToolTip(tooltip)

    def _handle_voice_state(self, msg) -> None:
        """Update voice menu label based on pipeline state."""
        state = msg.data.get("state", "idle")
        if state in ("listening", "recording", "processing", "speaking"):
            self._voice_action.setText(f"Voice: {state.capitalize()}")
        else:
            self._voice_action.setText("Voice: Start")

    def _toggle_voice(self) -> None:
        """Toggle voice pipeline via HTTP (fire-and-forget)."""
        import threading

        import httpx

        base = f"http://{self._settings.host}:{self._settings.port}"

        def _do_toggle():
            try:
                with httpx.Client(timeout=5.0) as client:
                    status_resp = client.get(f"{base}/voice/status")
                    is_running = status_resp.json().get("is_running", False)

                    if is_running:
                        client.post(f"{base}/voice/stop")
                    else:
                        client.post(f"{base}/voice/start")
            except Exception:
                logger.debug("Voice toggle failed", exc_info=True)

        threading.Thread(target=_do_toggle, daemon=True).start()

    def _open_viewport(self) -> None:
        """Open or show the noVNC viewport window."""
        if self._viewport_window is None:
            from alchemy.voice.tray.viewport import ViewportWindow

            self._viewport_window = ViewportWindow(
                novnc_url=self._settings.tray_novnc_url,
            )
        self._viewport_window.show()
        self._viewport_window.raise_()
        self._viewport_window.activateWindow()

    def _quit(self) -> None:
        """Clean shutdown."""
        if self._viewport_window:
            self._viewport_window.close()
        self._app.quit()
