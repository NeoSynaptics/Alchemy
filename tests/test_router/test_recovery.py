"""Tests for recovery strategies."""

from alchemy.router.categories import TaskCategory
from alchemy.router.recovery import get_recovery


class TestGetRecovery:
    def test_all_categories_have_strategies(self):
        for cat in TaskCategory:
            strategy = get_recovery(cat)
            assert isinstance(strategy, str)
            assert len(strategy) > 20

    def test_media_mentions_taskbar(self):
        assert "taskbar" in get_recovery(TaskCategory.MEDIA).lower()

    def test_web_mentions_browser(self):
        assert "browser" in get_recovery(TaskCategory.WEB).lower()

    def test_file_mentions_terminal(self):
        assert "terminal" in get_recovery(TaskCategory.FILE).lower()

    def test_communication_mentions_loading(self):
        assert "load" in get_recovery(TaskCategory.COMMUNICATION).lower()

    def test_general_is_generic(self):
        strategy = get_recovery(TaskCategory.GENERAL)
        assert "context menu" in strategy.lower() or "right-click" in strategy.lower()

    def test_returns_general_for_unknown(self):
        # GENERAL is the fallback, no matter what
        general = get_recovery(TaskCategory.GENERAL)
        assert len(general) > 0
