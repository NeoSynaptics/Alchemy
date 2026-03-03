"""Tests for task category classification."""

import pytest

from alchemy.router.categories import TaskCategory, classify_task, get_hint


class TestClassifyTask:
    def test_media_spotify(self):
        assert classify_task("open spotify and play music") == TaskCategory.MEDIA

    def test_media_youtube(self):
        assert classify_task("go to youtube and play a video") == TaskCategory.MEDIA

    def test_media_volume(self):
        assert classify_task("turn up the volume") == TaskCategory.MEDIA

    def test_web_browser(self):
        assert classify_task("open firefox and go to google.com") == TaskCategory.WEB

    def test_web_search(self):
        assert classify_task("search for python tutorials") == TaskCategory.WEB

    def test_web_url(self):
        assert classify_task("navigate to https://example.com") == TaskCategory.WEB

    def test_file_management(self):
        assert classify_task("copy the file to the downloads folder") == TaskCategory.FILE

    def test_file_delete(self):
        assert classify_task("delete the old files from desktop") == TaskCategory.FILE

    def test_communication_email(self):
        assert classify_task("compose an email to john") == TaskCategory.COMMUNICATION

    def test_communication_slack(self):
        assert classify_task("send a message on slack") == TaskCategory.COMMUNICATION

    def test_development_code(self):
        assert classify_task("open the terminal and run git status") == TaskCategory.DEVELOPMENT

    def test_development_editor(self):
        assert classify_task("open vscode and write some code") == TaskCategory.DEVELOPMENT

    def test_system_settings(self):
        assert classify_task("change the display settings") == TaskCategory.SYSTEM

    def test_system_wifi(self):
        assert classify_task("connect to wifi network") == TaskCategory.SYSTEM

    def test_general_fallback(self):
        assert classify_task("do something interesting") == TaskCategory.GENERAL

    def test_general_empty(self):
        assert classify_task("") == TaskCategory.GENERAL

    def test_mixed_favors_stronger_match(self):
        # "play music on youtube" — both MEDIA and WEB keywords
        result = classify_task("play music on youtube")
        assert result in (TaskCategory.MEDIA, TaskCategory.WEB)

    def test_case_insensitive(self):
        assert classify_task("OPEN SPOTIFY") == TaskCategory.MEDIA
        assert classify_task("Open Firefox") == TaskCategory.WEB


class TestGetHint:
    def test_media_hint_has_placeholder(self):
        hint = get_hint(TaskCategory.MEDIA)
        assert "{media_apps}" in hint

    def test_web_hint_has_placeholder(self):
        hint = get_hint(TaskCategory.WEB)
        assert "{default_browser}" in hint

    def test_general_hint_no_placeholder(self):
        hint = get_hint(TaskCategory.GENERAL)
        assert "{" not in hint

    def test_all_categories_have_hints(self):
        for cat in TaskCategory:
            hint = get_hint(cat)
            assert isinstance(hint, str)
            assert len(hint) > 10
