"""Tests for context-aware tier classification."""

import pytest

from alchemy.router.categories import TaskCategory
from alchemy.router.tier import classify_tier_contextual
from alchemy.schemas import ActionTier, VisionAction


def _action(action: str = "click", text: str | None = None, reasoning: str = "") -> VisionAction:
    return VisionAction(action=action, x=500, y=500, text=text, reasoning=reasoning)


class TestClassifyTierContextual:
    # --- Destructive patterns ---

    def test_destructive_text_gets_approve(self):
        action = _action("click", reasoning="I will delete this file")
        assert classify_tier_contextual(action, TaskCategory.FILE, "delete file") == ActionTier.APPROVE

    def test_type_delete_command_gets_approve(self):
        action = _action("type", text="delete all the old records")
        assert classify_tier_contextual(action, TaskCategory.DEVELOPMENT, "clean up") == ActionTier.APPROVE

    def test_uninstall_gets_approve(self):
        action = _action("click", reasoning="Clicking uninstall button to remove the app")
        assert classify_tier_contextual(action, TaskCategory.SYSTEM, "remove app") == ActionTier.APPROVE

    # --- Purchase patterns ---

    def test_purchase_click_gets_approve(self):
        action = _action("click", reasoning="Clicking the buy button")
        assert classify_tier_contextual(action, TaskCategory.WEB, "buy a book") == ActionTier.APPROVE

    def test_purchase_goal_with_click_gets_approve(self):
        action = _action("click")
        assert classify_tier_contextual(action, TaskCategory.WEB, "purchase the item") == ActionTier.APPROVE

    # --- Communication ---

    def test_communication_type_gets_notify(self):
        action = _action("type", text="Hello, how are you?")
        assert classify_tier_contextual(action, TaskCategory.COMMUNICATION, "send a message") == ActionTier.NOTIFY

    def test_communication_enter_gets_approve(self):
        action = _action("hotkey", text="Enter")
        assert classify_tier_contextual(action, TaskCategory.COMMUNICATION, "send a message on slack") == ActionTier.APPROVE

    def test_communication_ctrl_enter_gets_approve(self):
        action = _action("hotkey", text="ctrl+enter")
        assert classify_tier_contextual(action, TaskCategory.COMMUNICATION, "reply to email") == ActionTier.APPROVE

    # --- File operations ---

    def test_file_delete_hotkey_gets_approve(self):
        action = _action("hotkey", text="Delete")
        assert classify_tier_contextual(action, TaskCategory.FILE, "manage files") == ActionTier.APPROVE

    def test_file_normal_click_stays_auto(self):
        action = _action("click")
        assert classify_tier_contextual(action, TaskCategory.FILE, "open a folder") == ActionTier.AUTO

    # --- Fallback to base rules ---

    def test_media_click_stays_auto(self):
        action = _action("click")
        assert classify_tier_contextual(action, TaskCategory.MEDIA, "play music") == ActionTier.AUTO

    def test_type_in_non_communication_gets_notify(self):
        # Base classify_tier returns NOTIFY for type actions
        action = _action("type", text="hello world")
        result = classify_tier_contextual(action, TaskCategory.WEB, "search for something")
        assert result == ActionTier.NOTIFY

    def test_wait_stays_auto(self):
        action = _action("wait")
        assert classify_tier_contextual(action, TaskCategory.GENERAL, "anything") == ActionTier.AUTO

    def test_scroll_stays_auto(self):
        action = _action("scroll")
        assert classify_tier_contextual(action, TaskCategory.WEB, "browse page") == ActionTier.AUTO

    # --- Edge cases ---

    def test_no_text_doesnt_crash(self):
        action = _action("click", text=None, reasoning="")
        result = classify_tier_contextual(action, TaskCategory.GENERAL, "do something")
        assert result == ActionTier.AUTO

    def test_empty_goal_doesnt_crash(self):
        action = _action("click")
        result = classify_tier_contextual(action, TaskCategory.GENERAL, "")
        assert result == ActionTier.AUTO
