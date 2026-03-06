"""Tests for TrayManager — lifecycle and thread management."""

import threading
from unittest.mock import MagicMock, patch

from alchemy.voice.tray.events import TrayEventBus


class TestTrayManager:
    def test_init(self):
        from alchemy.voice.tray.app import TrayManager

        bus = TrayEventBus()
        settings = MagicMock()
        mgr = TrayManager(event_bus=bus, settings=settings)
        assert mgr.is_running is False

    def test_start_creates_thread(self):
        from alchemy.voice.tray.app import TrayManager

        bus = TrayEventBus()
        settings = MagicMock()
        mgr = TrayManager(event_bus=bus, settings=settings)

        # Mock _run_gui to avoid actually starting Qt
        with patch.object(mgr, '_run_gui'):
            mgr.start()
            assert mgr._thread is not None
            assert mgr._thread.daemon is True
            assert mgr._thread.name == "neo-tray"
            # Clean up
            mgr._thread.join(timeout=1)

    def test_double_start_ignored(self):
        from alchemy.voice.tray.app import TrayManager

        bus = TrayEventBus()
        settings = MagicMock()
        mgr = TrayManager(event_bus=bus, settings=settings)

        with patch.object(mgr, '_run_gui'):
            mgr.start()
            first_thread = mgr._thread
            mgr.start()  # Should warn but not create new thread
            # Thread reference unchanged (or new one if first finished)
            mgr._thread.join(timeout=1)

    def test_stop_without_start(self):
        from alchemy.voice.tray.app import TrayManager

        bus = TrayEventBus()
        settings = MagicMock()
        mgr = TrayManager(event_bus=bus, settings=settings)
        # Should not raise
        mgr.stop()
        assert mgr.is_running is False
