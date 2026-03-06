"""Approval dialog — screenshot preview + approve/deny + countdown.

Shown when Alchemy requests user approval for an APPROVE-tier action.
Auto-denies on timeout. Resolves the TrayMessage Future on close.
"""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

if TYPE_CHECKING:
    from alchemy.voice.tray.events import TrayEventBus, TrayMessage

logger = logging.getLogger(__name__)


class ApprovalDialog(QDialog):
    """Modal dialog for APPROVE-tier actions with countdown timer."""

    def __init__(
        self,
        msg: TrayMessage,
        event_bus: TrayEventBus,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._msg = msg
        self._event_bus = event_bus
        self._responded = False

        data = msg.data
        self._timeout = data.get("timeout_seconds", 60)
        self._remaining = self._timeout

        self.setWindowTitle("Neo — Approval Required")
        self.setMinimumWidth(480)
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowStaysOnTopHint
        )

        self._build_ui(data)
        self._start_countdown()

    def _build_ui(self, data: dict) -> None:
        """Build the dialog layout."""
        layout = QVBoxLayout(self)

        # Goal
        goal = data.get("goal", "Unknown task")
        goal_label = QLabel(f"<b>Task:</b> {goal}")
        goal_label.setWordWrap(True)
        layout.addWidget(goal_label)

        # Action info
        action = data.get("action", {})
        action_type = action.get("action", "unknown")
        reasoning = action.get("reasoning", "")
        action_text = f"<b>Action:</b> {action_type}"
        if reasoning:
            action_text += f" — {reasoning}"
        action_label = QLabel(action_text)
        action_label.setWordWrap(True)
        layout.addWidget(action_label)

        # Screenshot preview
        screenshot_b64 = data.get("screenshot_b64", "")
        if screenshot_b64:
            try:
                img_data = base64.b64decode(screenshot_b64)
                pixmap = QPixmap()
                pixmap.loadFromData(img_data)
                if not pixmap.isNull():
                    scaled = pixmap.scaledToWidth(
                        460, Qt.TransformationMode.SmoothTransformation
                    )
                    img_label = QLabel()
                    img_label.setPixmap(scaled)
                    img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    layout.addWidget(img_label)
            except Exception:
                logger.debug("Failed to decode screenshot", exc_info=True)

        # Step info
        step = data.get("step", 0)
        step_label = QLabel(f"Step: {step}")
        layout.addWidget(step_label)

        # Countdown
        self._countdown_label = QLabel(f"Auto-deny in {self._remaining}s")
        self._countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._countdown_label)

        # Buttons
        btn_layout = QHBoxLayout()
        self._approve_btn = QPushButton("Approve")
        self._approve_btn.setStyleSheet(
            "QPushButton { background-color: #2d7d46; color: white; "
            "padding: 8px 24px; font-weight: bold; }"
        )
        self._approve_btn.clicked.connect(self._on_approve)

        self._deny_btn = QPushButton("Deny")
        self._deny_btn.setStyleSheet(
            "QPushButton { background-color: #c0392b; color: white; "
            "padding: 8px 24px; font-weight: bold; }"
        )
        self._deny_btn.clicked.connect(self._on_deny)

        btn_layout.addWidget(self._approve_btn)
        btn_layout.addWidget(self._deny_btn)
        layout.addLayout(btn_layout)

    def _start_countdown(self) -> None:
        """Start the auto-deny countdown timer."""
        self._countdown_timer = QTimer(self)
        self._countdown_timer.timeout.connect(self._tick)
        self._countdown_timer.start(1000)

    def _tick(self) -> None:
        """Update countdown every second."""
        self._remaining -= 1
        self._countdown_label.setText(f"Auto-deny in {self._remaining}s")

        if self._remaining <= 0:
            self._countdown_timer.stop()
            self._respond(approved=False)
            self.close()

    def _on_approve(self) -> None:
        self._respond(approved=True)
        self.close()

    def _on_deny(self) -> None:
        self._respond(approved=False)
        self.close()

    def _respond(self, approved: bool) -> None:
        """Send response back through the event bus (once only)."""
        if self._responded:
            return
        self._responded = True
        self._event_bus.respond(self._msg, approved=approved)
        logger.info("Approval dialog: %s", "approved" if approved else "denied")

    def closeEvent(self, event) -> None:
        """Ensure we respond even if dialog is closed via X button."""
        if not self._responded:
            self._respond(approved=False)
        self._countdown_timer.stop()
        super().closeEvent(event)
