"""Tests for intent classifier — keyword matching and tag parsing."""

from alchemy.voice.models.schemas import RouteIntent
from alchemy.voice.router.classifier import classify_from_keywords, parse_intent_tag


class TestKeywordClassifier:
    # --- GUI task signals ---

    def test_open_app(self):
        assert classify_from_keywords("open Chrome") == RouteIntent.GUI_TASK

    def test_click(self):
        assert classify_from_keywords("click the submit button") == RouteIntent.GUI_TASK

    def test_send_email(self):
        assert classify_from_keywords("send email to John") == RouteIntent.GUI_TASK

    def test_navigate(self):
        assert classify_from_keywords("go to gmail.com") == RouteIntent.GUI_TASK

    def test_download(self):
        assert classify_from_keywords("download the PDF") == RouteIntent.GUI_TASK

    def test_screenshot(self):
        assert classify_from_keywords("take a screenshot") == RouteIntent.GUI_TASK

    def test_log_in(self):
        assert classify_from_keywords("log in to my account") == RouteIntent.GUI_TASK

    def test_purchase(self):
        assert classify_from_keywords("purchase the item") == RouteIntent.GUI_TASK

    # --- System command signals ---

    def test_pause(self):
        assert classify_from_keywords("pause") == RouteIntent.SYSTEM_COMMAND

    def test_status(self):
        assert classify_from_keywords("status") == RouteIntent.SYSTEM_COMMAND

    def test_stop_listening(self):
        assert classify_from_keywords("stop listening") == RouteIntent.SYSTEM_COMMAND

    def test_settings(self):
        assert classify_from_keywords("settings") == RouteIntent.SYSTEM_COMMAND

    # --- Ambiguous (should return None) ---

    def test_general_question(self):
        assert classify_from_keywords("what is Python?") is None

    def test_math(self):
        assert classify_from_keywords("what's 2+2?") is None

    def test_advice(self):
        assert classify_from_keywords("tell me about photosynthesis") is None

    def test_empty_string(self):
        assert classify_from_keywords("") is None

    # --- Case insensitivity ---

    def test_case_insensitive(self):
        assert classify_from_keywords("OPEN CHROME") == RouteIntent.GUI_TASK

    def test_mixed_case(self):
        assert classify_from_keywords("Send Email to work") == RouteIntent.GUI_TASK


class TestIntentTagParsing:
    def test_conversation_tag(self):
        intent, text = parse_intent_tag("[CONVERSATION]\nHere is my answer.")
        assert intent == RouteIntent.CONVERSATION
        assert text == "Here is my answer."

    def test_gui_task_tag(self):
        intent, text = parse_intent_tag("[GUI_TASK]\nI'll open that for you.")
        assert intent == RouteIntent.GUI_TASK
        assert text == "I'll open that for you."

    def test_system_tag(self):
        intent, text = parse_intent_tag("[SYSTEM]\nPausing now.")
        assert intent == RouteIntent.SYSTEM_COMMAND
        assert text == "Pausing now."

    def test_no_tag_returns_unclear(self):
        intent, text = parse_intent_tag("Just a normal response.")
        assert intent == RouteIntent.UNCLEAR
        assert text == "Just a normal response."

    def test_case_insensitive(self):
        intent, _ = parse_intent_tag("[conversation]\nanswer here")
        assert intent == RouteIntent.CONVERSATION

    def test_whitespace_before_tag(self):
        intent, text = parse_intent_tag("  [GUI_TASK]\nDoing it.")
        assert intent == RouteIntent.GUI_TASK
        assert text == "Doing it."

    def test_tag_without_newline(self):
        intent, text = parse_intent_tag("[CONVERSATION] answer here")
        assert intent == RouteIntent.CONVERSATION
        assert "answer here" in text

    def test_empty_string(self):
        intent, text = parse_intent_tag("")
        assert intent == RouteIntent.UNCLEAR
        assert text == ""

    def test_partial_tag_not_matched(self):
        intent, text = parse_intent_tag("[CONVER")
        assert intent == RouteIntent.UNCLEAR

    def test_unknown_tag_not_matched(self):
        intent, text = parse_intent_tag("[UNKNOWN]\ntext")
        assert intent == RouteIntent.UNCLEAR
