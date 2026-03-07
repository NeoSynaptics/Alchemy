"""QR-based device pairing — locks a phone to this Alchemy instance.

Flow:
  1. PC calls generate_qr() → gets QR data (server URL + token)
  2. Phone scans QR → stores token in SecureStore
  3. Phone connects to /ws/connect, sends system:auth with token
  4. Server verifies token → tunnel is live

Tokens are stored in SQLite. Each device gets a unique token.
Multiple phones per PC are supported, each tunnel is isolated.
"""

from __future__ import annotations

import json
import logging
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PairedDevice:
    device_id: str
    token: str
    device_name: str
    paired_at: float
    last_seen: float


class PairingManager:
    """Manages QR pairing and device token registry."""

    def __init__(self, data_dir: str | Path) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._data_dir / "devices.db"
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS devices (
                    device_id TEXT PRIMARY KEY,
                    token TEXT UNIQUE NOT NULL,
                    device_name TEXT NOT NULL DEFAULT '',
                    paired_at REAL NOT NULL,
                    last_seen REAL NOT NULL
                )
            """)
            conn.commit()

    def generate_qr_data(
        self,
        server_url: str,
        device_name: str = "",
    ) -> dict:
        """Generate QR payload for a new device pairing.

        Returns dict with: server, token, device_name, device_id
        The caller renders this as a QR code.
        """
        token = secrets.token_urlsafe(48)  # 256-bit
        device_id = secrets.token_hex(8)
        now = time.time()

        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO devices (device_id, token, device_name, paired_at, last_seen) "
                "VALUES (?, ?, ?, ?, ?)",
                (device_id, token, device_name or "phone", now, now),
            )
            conn.commit()

        qr_data = {
            "server": server_url,
            "token": token,
            "device_id": device_id,
            "device_name": "Alchemy-PC",
        }
        logger.info("Pairing QR generated for device_id=%s", device_id)
        return qr_data

    def verify_token(self, token: str) -> PairedDevice | None:
        """Verify a token and return the paired device, or None."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT device_id, token, device_name, paired_at, last_seen "
                "FROM devices WHERE token = ?",
                (token,),
            ).fetchone()

        if not row:
            return None

        # Update last_seen
        now = time.time()
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "UPDATE devices SET last_seen = ? WHERE token = ?",
                (now, token),
            )
            conn.commit()

        return PairedDevice(
            device_id=row[0],
            token=row[1],
            device_name=row[2],
            paired_at=row[3],
            last_seen=now,
        )

    def list_devices(self) -> list[PairedDevice]:
        """List all paired devices."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT device_id, token, device_name, paired_at, last_seen "
                "FROM devices ORDER BY paired_at DESC"
            ).fetchall()

        return [
            PairedDevice(
                device_id=r[0], token=r[1], device_name=r[2],
                paired_at=r[3], last_seen=r[4],
            )
            for r in rows
        ]

    def revoke_device(self, device_id: str) -> bool:
        """Revoke a paired device. Returns True if found and deleted."""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM devices WHERE device_id = ?", (device_id,),
            )
            conn.commit()
            deleted = cursor.rowcount > 0

        if deleted:
            logger.info("Device revoked: %s", device_id)
        return deleted

    def qr_to_json(self, qr_data: dict) -> str:
        """Serialize QR data to JSON string for QR code generation."""
        return json.dumps(qr_data, separators=(",", ":"))
