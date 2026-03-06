"""Tests for ConversationManager — context window, history, system prompt."""

from uuid import uuid4

from alchemy.voice.models.conversation import ConversationManager, DEFAULT_SYSTEM_PROMPT


class TestBasicOperations:
    def test_add_user_message(self):
        mgr = ConversationManager()
        cid = uuid4()
        msg = mgr.add_user_message(cid, "hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_add_assistant_message(self):
        mgr = ConversationManager()
        cid = uuid4()
        msg = mgr.add_assistant_message(cid, "hi back")
        assert msg.role == "assistant"
        assert msg.content == "hi back"

    def test_get_messages_includes_system_prompt(self):
        mgr = ConversationManager()
        cid = uuid4()
        mgr.add_user_message(cid, "test")
        messages = mgr.get_messages(cid)
        assert messages[0].role == "system"
        assert messages[0].content == DEFAULT_SYSTEM_PROMPT
        assert messages[1].role == "user"
        assert messages[1].content == "test"

    def test_custom_system_prompt(self):
        mgr = ConversationManager(system_prompt="Custom prompt")
        cid = uuid4()
        messages = mgr.get_messages(cid)
        assert messages[0].content == "Custom prompt"

    def test_system_prompt_property(self):
        mgr = ConversationManager(system_prompt="Test")
        assert mgr.system_prompt == "Test"

    def test_clear_conversation(self):
        mgr = ConversationManager()
        cid = uuid4()
        mgr.add_user_message(cid, "hello")
        mgr.clear(cid)
        messages = mgr.get_messages(cid)
        # Only system prompt remains
        assert len(messages) == 1

    def test_active_conversations(self):
        mgr = ConversationManager()
        assert mgr.active_conversations() == 0
        cid1 = uuid4()
        cid2 = uuid4()
        mgr.add_user_message(cid1, "a")
        mgr.add_user_message(cid2, "b")
        assert mgr.active_conversations() == 2


class TestSlidingWindow:
    def test_max_history_trimming(self):
        mgr = ConversationManager(max_history=3)
        cid = uuid4()
        for i in range(5):
            mgr.add_user_message(cid, f"msg-{i}")
        messages = mgr.get_messages(cid)
        # system + last 3 messages
        assert len(messages) == 4
        assert messages[1].content == "msg-2"

    def test_token_budget_trimming(self):
        mgr = ConversationManager(max_tokens_estimate=50)  # Very small budget
        cid = uuid4()
        # Each message ~200 chars (50 tokens). Budget = 50 tokens = 200 chars.
        mgr.add_user_message(cid, "x" * 200)
        mgr.add_user_message(cid, "y" * 200)
        mgr.add_user_message(cid, "z" * 200)
        messages = mgr.get_messages(cid)
        # Should trim oldest to fit. At minimum: system + 1 message
        assert len(messages) >= 2
        assert messages[0].role == "system"
