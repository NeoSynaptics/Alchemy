"""Tests for ConversationManager SQLite persistence.

Scaffold: ConversationManager.db_path not yet implemented.
These tests define the expected contract — implement, then remove xfail.
"""

import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from alchemy.voice.models.conversation import ConversationManager


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_conversations.db"


@pytest.fixture
def conv_id():
    return uuid4()


@pytest.mark.xfail(reason="db_path persistence not yet implemented")
def test_persist_and_reload(db_path, conv_id):
    """Messages persist to SQLite and reload on new instance."""
    mgr = ConversationManager(db_path=db_path)
    mgr.add_user_message(conv_id, "Hello")
    mgr.add_assistant_message(conv_id, "Hi there")

    # New manager loads from same DB
    mgr2 = ConversationManager(db_path=db_path)
    history = mgr2.get_context_window(conv_id)

    # system + 2 messages
    assert len(history) == 3
    assert history[1]["role"] == "user"
    assert history[1]["content"] == "Hello"
    assert history[2]["role"] == "assistant"
    assert history[2]["content"] == "Hi there"


@pytest.mark.xfail(reason="db_path persistence not yet implemented")
def test_clear_removes_from_db(db_path, conv_id):
    """Clearing a conversation removes it from SQLite too."""
    mgr = ConversationManager(db_path=db_path)
    mgr.add_user_message(conv_id, "Test")
    mgr.clear(conv_id)

    mgr2 = ConversationManager(db_path=db_path)
    history = mgr2.get_context_window(conv_id)
    # Only system prompt, no user messages
    assert len(history) == 1
    assert history[0]["role"] == "system"


@pytest.mark.xfail(reason="db_path persistence not yet implemented")
def test_no_db_path_still_works(conv_id):
    """Without db_path, ConversationManager works in-memory only."""
    mgr = ConversationManager()
    mgr.add_user_message(conv_id, "Hello")
    history = mgr.get_context_window(conv_id)
    assert len(history) == 2  # system + user


@pytest.mark.xfail(reason="db_path persistence not yet implemented")
def test_multiple_conversations_isolated(db_path):
    """Different conversation IDs stay isolated."""
    mgr = ConversationManager(db_path=db_path)
    c1, c2 = uuid4(), uuid4()

    mgr.add_user_message(c1, "Conv 1")
    mgr.add_user_message(c2, "Conv 2")

    mgr2 = ConversationManager(db_path=db_path)
    h1 = mgr2.get_context_window(c1)
    h2 = mgr2.get_context_window(c2)

    assert h1[1]["content"] == "Conv 1"
    assert h2[1]["content"] == "Conv 2"


@pytest.mark.xfail(reason="db_path persistence not yet implemented")
def test_db_creates_parent_dirs(tmp_path, conv_id):
    """DB init creates parent directories if needed."""
    deep_path = tmp_path / "a" / "b" / "c" / "conv.db"
    mgr = ConversationManager(db_path=deep_path)
    mgr.add_user_message(conv_id, "Deep test")
    assert deep_path.exists()
