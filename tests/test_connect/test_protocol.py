"""Tests for AlchemyConnect protocol — message envelope."""

import pytest

from alchemy.connect.protocol import AlchemyMessage, system_msg


class TestAlchemyMessage:
    def test_create_basic(self):
        msg = AlchemyMessage(agent="chat", type="message", payload={"text": "hi"})
        assert msg.agent == "chat"
        assert msg.type == "message"
        assert msg.payload == {"text": "hi"}
        assert msg.v == 1
        assert len(msg.id) == 12
        assert msg.ts > 0

    def test_to_dict_strips_none(self):
        msg = AlchemyMessage(agent="system", type="ping")
        d = msg.to_dict()
        assert "ref" not in d
        assert "seq" not in d
        assert d["agent"] == "system"
        assert d["type"] == "ping"

    def test_to_dict_keeps_ref(self):
        msg = AlchemyMessage(agent="chat", type="done", ref="abc123")
        d = msg.to_dict()
        assert d["ref"] == "abc123"

    def test_from_dict_basic(self):
        data = {"agent": "chat", "type": "message", "payload": {"text": "hello"}}
        msg = AlchemyMessage.from_dict(data)
        assert msg.agent == "chat"
        assert msg.type == "message"
        assert msg.payload["text"] == "hello"

    def test_from_dict_missing_agent(self):
        with pytest.raises(ValueError, match="agent"):
            AlchemyMessage.from_dict({"type": "message"})

    def test_from_dict_missing_type(self):
        with pytest.raises(ValueError, match="type"):
            AlchemyMessage.from_dict({"agent": "chat"})

    def test_from_dict_not_dict(self):
        with pytest.raises(ValueError, match="JSON object"):
            AlchemyMessage.from_dict("not a dict")

    def test_from_dict_preserves_optional(self):
        data = {
            "v": 1, "id": "custom_id", "agent": "system",
            "type": "hello", "ts": 12345, "ref": "ref_id", "seq": 42,
            "payload": {},
        }
        msg = AlchemyMessage.from_dict(data)
        assert msg.id == "custom_id"
        assert msg.ref == "ref_id"
        assert msg.seq == 42
        assert msg.ts == 12345

    def test_from_dict_defaults(self):
        msg = AlchemyMessage.from_dict({"agent": "x", "type": "y"})
        assert msg.v == 1
        assert msg.payload == {}
        assert msg.ref is None
        assert msg.seq is None

    def test_reply(self):
        original = AlchemyMessage(agent="chat", type="message", id="orig123")
        reply = original.reply("done", {"text": "response"})
        assert reply.agent == "chat"
        assert reply.type == "done"
        assert reply.ref == "orig123"
        assert reply.payload["text"] == "response"

    def test_roundtrip(self):
        msg = AlchemyMessage(
            agent="browser", type="scrape",
            payload={"url": "https://example.com"}, ref="ref1", seq=7,
        )
        d = msg.to_dict()
        restored = AlchemyMessage.from_dict(d)
        assert restored.agent == msg.agent
        assert restored.type == msg.type
        assert restored.payload == msg.payload
        assert restored.ref == msg.ref
        assert restored.seq == msg.seq


class TestSystemMsg:
    def test_creates_system_message(self):
        msg = system_msg("hello", {"version": "1.0"})
        assert msg.agent == "system"
        assert msg.type == "hello"
        assert msg.payload["version"] == "1.0"

    def test_default_empty_payload(self):
        msg = system_msg("ping")
        assert msg.payload == {}
