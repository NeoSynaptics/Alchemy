"""ConnectionHub — WebSocket lifecycle for the AlchemyConnect tunnel.

Handles: accept, auth, routing, keepalive, offline queue drain, cleanup.
One hub instance per Alchemy server.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import WebSocket, WebSocketDisconnect

from alchemy.connect.pairing import PairedDevice, PairingManager
from alchemy.connect.protocol import AlchemyMessage, system_msg
from alchemy.connect.queue import OfflineQueue
from alchemy.connect.router import AgentRouter

logger = logging.getLogger(__name__)


@dataclass
class DeviceConnection:
    """A live WebSocket connection from a paired device."""

    device_id: str
    device_name: str
    ws: WebSocket
    connected_at: float = field(default_factory=time.time)
    last_ping: float = field(default_factory=time.time)


class ConnectionHub:
    """Manages WebSocket connections and message routing."""

    def __init__(
        self,
        pairing: PairingManager,
        router: AgentRouter,
        queue: OfflineQueue,
        auth_timeout: float = 10.0,
        ping_interval: float = 30.0,
        server_version: str = "0.1.0",
    ) -> None:
        self._pairing = pairing
        self._router = router
        self._queue = queue
        self._auth_timeout = auth_timeout
        self._ping_interval = ping_interval
        self._server_version = server_version
        self._connections: dict[str, DeviceConnection] = {}
        self._ping_tasks: dict[str, asyncio.Task] = {}
        self._gpu_semaphore = asyncio.Semaphore(2)  # Dual GPU — 2 concurrent ops

    @asynccontextmanager
    async def gpu_guard(self):
        """Acquire GPU semaphore for heavy operations (image gen, etc.)."""
        await self._gpu_semaphore.acquire()
        try:
            yield
        finally:
            self._gpu_semaphore.release()

    @property
    def connected_devices(self) -> int:
        return len(self._connections)

    def get_connection(self, device_id: str) -> DeviceConnection | None:
        return self._connections.get(device_id)

    async def handle_websocket(self, ws: WebSocket) -> None:
        """Main WebSocket handler — accept, auth, message loop, cleanup."""
        await ws.accept()

        # Send hello
        hello = system_msg("hello", {
            "session_id": id(ws),
            "server_version": self._server_version,
            "available_agents": self._router.available_agents,
        })
        await self._send(ws, hello)

        # Wait for auth
        device = await self._authenticate(ws)
        if not device:
            return

        conn = DeviceConnection(
            device_id=device.device_id,
            device_name=device.device_name,
            ws=ws,
        )

        # Register connection (disconnect previous if same device)
        old = self._connections.get(device.device_id)
        if old:
            logger.info("Device %s reconnecting, closing old connection", device.device_id)
            try:
                await old.ws.close(code=4001, reason="Replaced by new connection")
            except Exception:
                pass
            self._cancel_ping(device.device_id)

        self._connections[device.device_id] = conn
        logger.info(
            "Device connected: %s (%s) — %d total",
            device.device_id, device.device_name, len(self._connections),
        )

        # Send auth_ok
        await self._send(ws, system_msg("auth_ok", {
            "device_name": device.device_name,
            "paired_at": device.paired_at,
        }))

        # Drain offline queue
        queued = self._queue.drain(device.device_id)
        if queued:
            logger.info("Draining %d queued messages for %s", len(queued), device.device_id)
            for msg in queued:
                await self._send(ws, msg)

        # Start keepalive
        self._ping_tasks[device.device_id] = asyncio.create_task(
            self._ping_loop(conn)
        )

        # Message loop
        try:
            await self._message_loop(conn)
        except WebSocketDisconnect:
            logger.info("Device disconnected: %s", device.device_id)
        except Exception:
            logger.exception("Error in connection loop for %s", device.device_id)
        finally:
            self._cleanup(device.device_id)

    async def _authenticate(self, ws: WebSocket) -> PairedDevice | None:
        """Wait for system:auth message within timeout."""
        try:
            raw = await asyncio.wait_for(ws.receive_text(), timeout=self._auth_timeout)
            data = json.loads(raw)
            msg = AlchemyMessage.from_dict(data)
        except asyncio.TimeoutError:
            await self._send(ws, system_msg("auth_fail", {"reason": "Auth timeout"}))
            await ws.close(code=4000, reason="Auth timeout")
            return None
        except (json.JSONDecodeError, ValueError) as e:
            await self._send(ws, system_msg("auth_fail", {"reason": str(e)}))
            await ws.close(code=4000, reason="Invalid auth message")
            return None

        if msg.agent != "system" or msg.type != "auth":
            await self._send(ws, system_msg("auth_fail", {"reason": "Expected system:auth"}))
            await ws.close(code=4000, reason="Expected auth")
            return None

        token = msg.payload.get("token", "")
        device = self._pairing.verify_token(token)
        if not device:
            await self._send(ws, system_msg("auth_fail", {"reason": "Invalid token"}))
            await ws.close(code=4001, reason="Invalid token")
            return None

        return device

    async def _message_loop(self, conn: DeviceConnection) -> None:
        """Process incoming messages from an authenticated device."""
        while True:
            raw = await conn.ws.receive_text()
            try:
                data = json.loads(raw)
                msg = AlchemyMessage.from_dict(data)
            except (json.JSONDecodeError, ValueError) as e:
                await self._send(conn.ws, system_msg("error", {
                    "reason": f"Invalid message: {e}",
                }))
                continue

            # System messages
            if msg.agent == "system":
                await self._handle_system(conn, msg)
                continue

            # Route to agent
            agent = self._router.get(msg.agent)
            if not agent:
                await self._send(conn.ws, system_msg("error", {
                    "reason": f"Unknown agent: {msg.agent}",
                }))
                continue

            try:
                async for response in agent.handle(msg, conn.device_id):
                    await self._send(conn.ws, response)
            except Exception:
                logger.exception("Agent %s error handling message", msg.agent)
                await self._send(conn.ws, AlchemyMessage(
                    agent=msg.agent,
                    type="error",
                    payload={"reason": "Internal agent error"},
                    ref=msg.id,
                ))

    async def _handle_system(self, conn: DeviceConnection, msg: AlchemyMessage) -> None:
        """Handle system-level messages (ping, agents discovery)."""
        if msg.type == "ping":
            conn.last_ping = time.time()
            await self._send(conn.ws, system_msg("pong", {"ts": time.time() * 1000}))
        elif msg.type == "pong":
            conn.last_ping = time.time()
        elif msg.type == "agents":
            await self._send(conn.ws, system_msg("agents", {
                "agents": self._router.describe_all(),
            }))
        else:
            await self._send(conn.ws, system_msg("error", {
                "reason": f"Unknown system type: {msg.type}",
            }))

    async def _ping_loop(self, conn: DeviceConnection) -> None:
        """Send periodic pings to keep the connection alive."""
        try:
            while True:
                await asyncio.sleep(self._ping_interval)
                try:
                    await self._send(conn.ws, system_msg("ping", {
                        "ts": time.time() * 1000,
                    }))
                except Exception:
                    break
        except asyncio.CancelledError:
            pass

    def _cancel_ping(self, device_id: str) -> None:
        task = self._ping_tasks.pop(device_id, None)
        if task and not task.done():
            task.cancel()

    def _cleanup(self, device_id: str) -> None:
        """Remove a device connection and cancel its ping task."""
        self._connections.pop(device_id, None)
        self._cancel_ping(device_id)
        logger.info("Device cleaned up: %s — %d remaining", device_id, len(self._connections))

    async def push(self, device_id: str, msg: AlchemyMessage) -> bool:
        """Push a message to a specific device. Queues if offline.

        Returns True if sent immediately, False if queued.
        """
        conn = self._connections.get(device_id)
        if conn:
            try:
                await self._send(conn.ws, msg)
                return True
            except Exception:
                logger.warning("Failed to push to %s, queueing", device_id)

        self._queue.enqueue(device_id, msg)
        return False

    async def broadcast(self, msg: AlchemyMessage) -> int:
        """Push a message to ALL connected devices. Returns count sent."""
        sent = 0
        for conn in list(self._connections.values()):
            try:
                await self._send(conn.ws, msg)
                sent += 1
            except Exception:
                logger.warning("Broadcast failed for %s", conn.device_id)
        return sent

    async def disconnect_all(self) -> None:
        """Gracefully close all connections."""
        for conn in list(self._connections.values()):
            try:
                await conn.ws.close(code=1001, reason="Server shutting down")
            except Exception:
                pass
        self._connections.clear()
        for task in self._ping_tasks.values():
            task.cancel()
        self._ping_tasks.clear()

    @staticmethod
    async def _send(ws: WebSocket, msg: AlchemyMessage) -> None:
        await ws.send_text(json.dumps(msg.to_dict()))
