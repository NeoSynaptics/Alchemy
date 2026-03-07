"""AlchemyConnect — universal tunnel/bus for external apps to reach Alchemy.

This is the transport layer between phones (and future external apps) and
the Alchemy core. It provides:

  - QR-locked device pairing (phone scans QR, gets a permanent token)
  - WebSocket tunnel at /ws/connect with auth + keepalive
  - Agent-based message routing (messages target a ConnectAgent by name)
  - Offline queue (messages queued when device is disconnected)

Phone apps are SEPARATE from Alchemy. They connect through this tunnel
and call AlchemyAgents. Agents abstract all core internals — developers
never touch GPU models, VRAM, or core primitives directly.

Architecture:
    Phone App (separate repo)
        | QR-locked WebSocket
    AlchemyConnect (this module, Tier 1 infra)
        | Routes to...
    ConnectAgents (chat, browser, voice, flow)
        | Uses...
    AlchemyCore (APU, Voice, Click, Cloud AI)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket

from alchemy.connect.hub import ConnectionHub
from alchemy.connect.manifest import MANIFEST
from alchemy.connect.pairing import PairingManager
from alchemy.connect.protocol import AlchemyMessage
from alchemy.connect.queue import OfflineQueue
from alchemy.connect.router import AgentRouter, ConnectAgent

logger = logging.getLogger(__name__)


class AlchemyConnect:
    """Public facade for the AlchemyConnect tunnel.

    Usage in server lifespan:
        connect = AlchemyConnect(app, settings)
        await connect.start()
        # ... server runs ...
        await connect.stop()
    """

    def __init__(self, app: FastAPI, settings: Any) -> None:
        self._app = app
        self._settings = settings

        # Resolve data dir
        connect_settings = getattr(settings, "connect", None)
        if connect_settings:
            data_dir = Path(connect_settings.data_dir)
            auth_timeout = connect_settings.auth_timeout_seconds
            ping_interval = connect_settings.ping_interval_seconds
            max_queue = connect_settings.offline_queue_max
        else:
            data_dir = Path("data/connect")
            auth_timeout = 10.0
            ping_interval = 30.0
            max_queue = 200

        self._pairing = PairingManager(data_dir=data_dir)
        self._router = AgentRouter()
        self._queue = OfflineQueue(data_dir=data_dir, max_per_device=max_queue)
        self._hub = ConnectionHub(
            pairing=self._pairing,
            router=self._router,
            queue=self._queue,
            auth_timeout=auth_timeout,
            ping_interval=ping_interval,
        )

    async def start(self) -> None:
        """Register the /ws/connect endpoint and pairing API routes."""
        from fastapi import APIRouter
        from fastapi.responses import JSONResponse

        # WebSocket endpoint
        @self._app.websocket("/ws/connect")
        async def ws_connect(ws: WebSocket):
            await self._hub.handle_websocket(ws)

        # Pairing REST endpoints
        pair_router = APIRouter(prefix="/v1/connect", tags=["connect"])

        @pair_router.post("/pair/qr")
        async def generate_qr():
            server_host = self._settings.host if hasattr(self._settings, "host") else "0.0.0.0"
            server_port = self._settings.port if hasattr(self._settings, "port") else 8000
            server_url = f"ws://{server_host}:{server_port}/ws/connect"
            qr_data = self._pairing.generate_qr_data(server_url=server_url)
            return JSONResponse(qr_data)

        @pair_router.get("/devices")
        async def list_devices():
            devices = self._pairing.list_devices()
            return [{
                "device_id": d.device_id,
                "device_name": d.device_name,
                "paired_at": d.paired_at,
                "last_seen": d.last_seen,
            } for d in devices]

        @pair_router.delete("/devices/{device_id}")
        async def revoke_device(device_id: str):
            ok = self._pairing.revoke_device(device_id)
            # Also disconnect if currently connected
            conn = self._hub.get_connection(device_id)
            if conn:
                try:
                    await conn.ws.close(code=4002, reason="Device revoked")
                except Exception:
                    pass
            return {"revoked": ok}

        @pair_router.get("/status")
        async def connect_status():
            return {
                "connected_devices": self._hub.connected_devices,
                "available_agents": self._router.available_agents,
                "agents": self._router.describe_all(),
            }

        self._app.include_router(pair_router)
        logger.info(
            "AlchemyConnect started — /ws/connect + %d agents",
            len(self._router.available_agents),
        )

    async def stop(self) -> None:
        """Gracefully close all connections."""
        await self._hub.disconnect_all()
        logger.info("AlchemyConnect stopped")

    # --- Public API for other Alchemy modules ---

    def register_agent(self, agent: ConnectAgent) -> None:
        """Register a ConnectAgent as callable through the tunnel."""
        self._router.register(agent)

    @property
    def available_agents(self) -> list[str]:
        return self._router.available_agents

    @property
    def connected_devices(self) -> int:
        return self._hub.connected_devices

    async def push(
        self,
        device_id: str,
        agent: str,
        type: str,
        payload: dict[str, Any] | None = None,
    ) -> bool:
        """Push a message to a specific device. Queues if offline."""
        msg = AlchemyMessage(agent=agent, type=type, payload=payload or {})
        return await self._hub.push(device_id, msg)

    async def broadcast(
        self,
        agent: str,
        type: str,
        payload: dict[str, Any] | None = None,
    ) -> int:
        """Push a message to ALL connected devices."""
        msg = AlchemyMessage(agent=agent, type=type, payload=payload or {})
        return await self._hub.broadcast(msg)

    @property
    def pairing(self) -> PairingManager:
        return self._pairing


__all__ = [
    "MANIFEST",
    "AlchemyConnect",
    "AlchemyMessage",
    "ConnectAgent",
    "AgentRouter",
]
