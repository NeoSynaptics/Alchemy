"""Tests for completion criteria."""

from alchemy.router.categories import TaskCategory
from alchemy.router.completion import get_completion


class TestGetCompletion:
    def test_all_categories_have_criteria(self):
        for cat in TaskCategory:
            criteria = get_completion(cat)
            assert isinstance(criteria, str)
            assert len(criteria) > 20

    def test_media_mentions_playing(self):
        assert "playing" in get_completion(TaskCategory.MEDIA).lower()

    def test_web_mentions_loaded(self):
        assert "loaded" in get_completion(TaskCategory.WEB).lower()

    def test_file_mentions_confirmed(self):
        assert "confirmed" in get_completion(TaskCategory.FILE).lower()

    def test_communication_mentions_sent(self):
        assert "sent" in get_completion(TaskCategory.COMMUNICATION).lower()

    def test_development_mentions_saved(self):
        criteria = get_completion(TaskCategory.DEVELOPMENT)
        assert "saved" in criteria.lower() or "finished" in criteria.lower()

    def test_general_mentions_finished(self):
        assert "finished" in get_completion(TaskCategory.GENERAL).lower()
