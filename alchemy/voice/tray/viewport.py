"""Viewport window — embeds Alchemy's shadow desktop via noVNC.

QWebEngineView loads the noVNC web client, providing a live view
of the WSL2 shadow desktop where Alchemy's vision agent operates.
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QWidget

logger = logging.getLogger(__name__)

_DEFAULT_NOVNC_URL = "http://localhost:6080/vnc.html?autoconnect=true&resize=scale"


class ViewportWindow(QMainWindow):
    """Window embedding the noVNC viewer for Alchemy's shadow desktop."""

    def __init__(
        self,
        novnc_url: str = _DEFAULT_NOVNC_URL,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._novnc_url = novnc_url

        self.setWindowTitle("Neo \u2014 Shadow Desktop")
        self.resize(1024, 768)

        self._build_ui()

    def _build_ui(self) -> None:
        """Create the web engine view."""
        try:
            from PyQt6.QtWebEngineWidgets import QWebEngineView

            central = QWidget()
            layout = QVBoxLayout(central)
            layout.setContentsMargins(0, 0, 0, 0)

            self._web_view = QWebEngineView()
            self._web_view.setUrl(QUrl(self._novnc_url))
            layout.addWidget(self._web_view)

            self.setCentralWidget(central)
        except ImportError:
            logger.warning(
                "PyQt6-WebEngine not installed — viewport disabled. "
                "Run: pip install PyQt6-WebEngine"
            )
            # Show placeholder
            placeholder = QWidget()
            layout = QVBoxLayout(placeholder)
            from PyQt6.QtWidgets import QLabel
            from PyQt6.QtCore import Qt

            label = QLabel("Viewport requires PyQt6-WebEngine\npip install PyQt6-WebEngine")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)
            self.setCentralWidget(placeholder)
            self._web_view = None

    def reload(self) -> None:
        """Reload the noVNC page."""
        if self._web_view:
            self._web_view.reload()
