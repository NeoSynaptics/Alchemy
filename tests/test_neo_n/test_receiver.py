"""Tests for NEO-N FileReceiver — file saving, validation, rate limiting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alchemy.neo_n.receiver import FileReceiver, BLOCKED_EXTENSIONS


@pytest.fixture
def inbox(tmp_path):
    return tmp_path / "neo_n_inbox"


@pytest.fixture
def receiver(inbox):
    return FileReceiver(
        inbox_path=inbox,
        max_file_size_mb=1,
        rate_limit_per_hour=5,
    )


class TestFileReceiver:
    async def test_save_basic(self, receiver, inbox):
        result = await receiver.save(
            file_data=b"hello world",
            original_filename="test.txt",
            content_type="text/plain",
            device_id="dev_001",
            device_name="Test Phone",
        )

        assert result.success
        assert result.size_bytes == 11
        assert "test" in result.filename
        assert result.filename.endswith(".txt")

        # File should exist in inbox
        saved = list(inbox.glob("*.txt"))
        assert len(saved) == 1

        # Sidecar should exist
        sidecars = list(inbox.glob("*.json"))
        assert len(sidecars) == 1
        sidecar = json.loads(sidecars[0].read_text())
        assert sidecar["device_id"] == "dev_001"
        assert sidecar["source_device"] == "Test Phone"
        assert sidecar["original_filename"] == "test.txt"
        assert sidecar["size_bytes"] == 11

    async def test_save_with_metadata(self, receiver, inbox):
        result = await receiver.save(
            file_data=b"photo",
            original_filename="IMG_001.jpg",
            content_type="image/jpeg",
            device_id="dev_001",
            device_name="iPhone",
            metadata={"tags": ["vacation", "beach"], "notes": "Sunset", "source_app": "Photos"},
        )

        assert result.success
        sidecars = list(inbox.glob("*.json"))
        sidecar = json.loads(sidecars[0].read_text())
        assert sidecar["user_tags"] == ["vacation", "beach"]
        assert sidecar["user_notes"] == "Sunset"
        assert sidecar["source_app"] == "Photos"

    async def test_blocked_extension(self, receiver):
        for ext in [".exe", ".bat", ".ps1", ".sh"]:
            result = await receiver.save(
                file_data=b"malicious",
                original_filename=f"bad{ext}",
                content_type="application/octet-stream",
                device_id="dev_001",
            )
            assert not result.success
            assert "not allowed" in result.error

    async def test_file_too_large(self, receiver):
        big_data = b"x" * (2 * 1024 * 1024)  # 2MB, limit is 1MB
        result = await receiver.save(
            file_data=big_data,
            original_filename="huge.zip",
            content_type="application/zip",
            device_id="dev_001",
        )
        assert not result.success
        assert "too large" in result.error

    async def test_rate_limit(self, receiver):
        for i in range(5):
            result = await receiver.save(
                file_data=b"ok",
                original_filename=f"file{i}.txt",
                content_type="text/plain",
                device_id="dev_rate",
            )
            assert result.success

        # 6th should be rate limited
        result = await receiver.save(
            file_data=b"blocked",
            original_filename="file5.txt",
            content_type="text/plain",
            device_id="dev_rate",
        )
        assert not result.success
        assert "Rate limit" in result.error

    async def test_rate_limit_per_device(self, receiver):
        """Different devices have independent rate limits."""
        for i in range(5):
            await receiver.save(
                file_data=b"ok",
                original_filename=f"file{i}.txt",
                content_type="text/plain",
                device_id="dev_a",
            )

        # dev_b should still be able to upload
        result = await receiver.save(
            file_data=b"ok",
            original_filename="file0.txt",
            content_type="text/plain",
            device_id="dev_b",
        )
        assert result.success

    async def test_mark_processed(self, receiver, inbox):
        result = await receiver.save(
            file_data=b"data",
            original_filename="photo.jpg",
            content_type="image/jpeg",
            device_id="dev_001",
        )
        assert result.success

        ok = receiver.mark_processed(result.filename)
        assert ok

        # File should be in processed/
        processed = list((inbox / "processed").glob("*.jpg"))
        assert len(processed) == 1

        # Sidecar should also be moved
        processed_sidecars = list((inbox / "processed").glob("*.json"))
        assert len(processed_sidecars) == 1

        # Original should be gone
        assert not (inbox / result.filename).exists()

    async def test_list_pending(self, receiver):
        await receiver.save(
            file_data=b"a", original_filename="a.txt",
            content_type="text/plain", device_id="dev_001", device_name="Phone",
        )
        await receiver.save(
            file_data=b"bb", original_filename="b.jpg",
            content_type="image/jpeg", device_id="dev_001", device_name="Phone",
        )

        pending = receiver.list_pending()
        assert len(pending) == 2
        assert all(p["source_device"] == "Phone" for p in pending)

    async def test_stats(self, receiver):
        await receiver.save(
            file_data=b"hello", original_filename="test.txt",
            content_type="text/plain", device_id="dev_001",
        )

        stats = receiver.stats()
        assert stats["pending_count"] == 1
        assert stats["pending_bytes"] == 5
        assert stats["processed_count"] == 0

    async def test_inbox_dirs_created(self, inbox):
        """Receiver should create inbox and processed dirs on init."""
        assert not inbox.exists()
        FileReceiver(inbox_path=inbox)
        assert inbox.exists()
        assert (inbox / "processed").exists()


class TestBlockedExtensions:
    def test_all_dangerous_extensions_blocked(self):
        dangerous = {".exe", ".bat", ".sh", ".ps1", ".cmd", ".com", ".scr", ".msi", ".vbs", ".wsf"}
        assert dangerous == BLOCKED_EXTENSIONS
