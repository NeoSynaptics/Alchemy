"""Tests for AlchemyConnect pairing — QR code device management."""

import json

import pytest

from alchemy.connect.pairing import PairingManager


@pytest.fixture
def pairing(tmp_path):
    return PairingManager(data_dir=tmp_path / "connect")


class TestPairingManager:
    def test_generate_qr_data(self, pairing):
        qr = pairing.generate_qr_data("ws://192.168.0.4:8000/ws/connect")
        assert qr["server"] == "ws://192.168.0.4:8000/ws/connect"
        assert len(qr["token"]) > 32
        assert len(qr["device_id"]) == 16
        assert qr["device_name"] == "Alchemy-PC"

    def test_verify_valid_token(self, pairing):
        qr = pairing.generate_qr_data("ws://localhost:8000/ws/connect")
        device = pairing.verify_token(qr["token"])
        assert device is not None
        assert device.device_id == qr["device_id"]

    def test_verify_invalid_token(self, pairing):
        assert pairing.verify_token("nonexistent_token") is None

    def test_verify_updates_last_seen(self, pairing):
        qr = pairing.generate_qr_data("ws://localhost:8000/ws/connect")
        d1 = pairing.verify_token(qr["token"])
        d2 = pairing.verify_token(qr["token"])
        assert d2.last_seen >= d1.last_seen

    def test_list_devices_empty(self, pairing):
        assert pairing.list_devices() == []

    def test_list_devices(self, pairing):
        pairing.generate_qr_data("ws://a")
        pairing.generate_qr_data("ws://b")
        devices = pairing.list_devices()
        assert len(devices) == 2

    def test_revoke_device(self, pairing):
        qr = pairing.generate_qr_data("ws://localhost:8000/ws/connect")
        assert pairing.revoke_device(qr["device_id"]) is True
        assert pairing.verify_token(qr["token"]) is None
        assert len(pairing.list_devices()) == 0

    def test_revoke_nonexistent(self, pairing):
        assert pairing.revoke_device("nonexistent") is False

    def test_multiple_devices_independent(self, pairing):
        qr1 = pairing.generate_qr_data("ws://a")
        qr2 = pairing.generate_qr_data("ws://b")
        pairing.revoke_device(qr1["device_id"])
        assert pairing.verify_token(qr1["token"]) is None
        assert pairing.verify_token(qr2["token"]) is not None

    def test_custom_device_name(self, pairing):
        qr = pairing.generate_qr_data("ws://a", device_name="my-iphone")
        device = pairing.verify_token(qr["token"])
        assert device.device_name == "my-iphone"

    def test_qr_to_json(self, pairing):
        qr = pairing.generate_qr_data("ws://localhost:8000/ws/connect")
        j = pairing.qr_to_json(qr)
        parsed = json.loads(j)
        assert parsed["server"] == qr["server"]
        assert parsed["token"] == qr["token"]
