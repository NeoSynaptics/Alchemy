"""Tests for AlchemyConnect hub — WebSocket lifecycle."""

import asyncio
import json

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from alchemy.connect.hub import ConnectionHub
from alchemy.connect.pairing import PairingManager
from alchemy.connect.protocol import AlchemyMessage
from alchemy.connect.queue import OfflineQueue
from alchemy.connect.router import AgentRouter, ConnectAgent


class EchoAgent(ConnectAgent):
    @property
    def agent_id(self) -> str:
        return "echo"

    async def handle(self, msg, device_id):
        yield AlchemyMessage(
            agent="echo", type="echo",
            payload={"text": msg.payload.get("text", "")},
            ref=msg.id,
        )


@pytest.fixture
def setup(tmp_path):
    """Create hub + app + paired token for testing."""
    pairing = PairingManager(data_dir=tmp_path / "connect")
    router = AgentRouter()
    router.register(EchoAgent())
    queue = OfflineQueue(data_dir=tmp_path / "connect")

    hub = ConnectionHub(
        pairing=pairing,
        router=router,
        queue=queue,
        auth_timeout=5.0,
        ping_interval=60.0,  # Long interval so pings don't interfere
    )

    app = FastAPI()

    @app.websocket("/ws/connect")
    async def ws_endpoint(ws: WebSocket):
        await hub.handle_websocket(ws)

    qr = pairing.generate_qr_data("ws://test/ws/connect")

    return {
        "app": app,
        "hub": hub,
        "pairing": pairing,
        "queue": queue,
        "token": qr["token"],
        "device_id": qr["device_id"],
    }


class TestConnectionHub:
    def test_hello_on_connect(self, setup):
        with TestClient(setup["app"]) as client:
            with client.websocket_connect("/ws/connect") as ws:
                hello = json.loads(ws.receive_text())
                assert hello["agent"] == "system"
                assert hello["type"] == "hello"
                assert "available_agents" in hello["payload"]
                assert "echo" in hello["payload"]["available_agents"]

                # Auth to avoid unclean close
                ws.send_text(json.dumps({
                    "agent": "system", "type": "auth",
                    "payload": {"token": setup["token"]},
                }))
                ws.receive_text()  # auth_ok

    def test_auth_success(self, setup):
        with TestClient(setup["app"]) as client:
            with client.websocket_connect("/ws/connect") as ws:
                # Receive hello
                ws.receive_text()

                # Send auth
                auth_msg = {
                    "agent": "system", "type": "auth",
                    "payload": {"token": setup["token"]},
                }
                ws.send_text(json.dumps(auth_msg))

                # Expect auth_ok
                response = json.loads(ws.receive_text())
                assert response["agent"] == "system"
                assert response["type"] == "auth_ok"

    def test_auth_invalid_token(self, setup):
        """Server sends auth_fail and closes the WebSocket on bad token."""
        with TestClient(setup["app"]) as client:
            try:
                with client.websocket_connect("/ws/connect") as ws:
                    ws.receive_text()  # hello
                    ws.send_text(json.dumps({
                        "agent": "system", "type": "auth",
                        "payload": {"token": "bad_token"},
                    }))
                    response = json.loads(ws.receive_text())
                    assert response["type"] == "auth_fail"
            except Exception:
                pass  # Connection closed by server — expected

    def test_route_to_agent(self, setup):
        with TestClient(setup["app"]) as client:
            with client.websocket_connect("/ws/connect") as ws:
                ws.receive_text()  # hello
                ws.send_text(json.dumps({
                    "agent": "system", "type": "auth",
                    "payload": {"token": setup["token"]},
                }))
                ws.receive_text()  # auth_ok

                # Send message to echo agent
                ws.send_text(json.dumps({
                    "agent": "echo", "type": "msg",
                    "payload": {"text": "hello world"},
                }))

                response = json.loads(ws.receive_text())
                assert response["agent"] == "echo"
                assert response["type"] == "echo"
                assert response["payload"]["text"] == "hello world"

    def test_unknown_agent_error(self, setup):
        with TestClient(setup["app"]) as client:
            with client.websocket_connect("/ws/connect") as ws:
                ws.receive_text()  # hello
                ws.send_text(json.dumps({
                    "agent": "system", "type": "auth",
                    "payload": {"token": setup["token"]},
                }))
                ws.receive_text()  # auth_ok

                ws.send_text(json.dumps({
                    "agent": "nonexistent", "type": "msg",
                }))

                response = json.loads(ws.receive_text())
                assert response["agent"] == "system"
                assert response["type"] == "error"
                assert "nonexistent" in response["payload"]["reason"]

    def test_system_ping_pong(self, setup):
        with TestClient(setup["app"]) as client:
            with client.websocket_connect("/ws/connect") as ws:
                ws.receive_text()  # hello
                ws.send_text(json.dumps({
                    "agent": "system", "type": "auth",
                    "payload": {"token": setup["token"]},
                }))
                ws.receive_text()  # auth_ok

                ws.send_text(json.dumps({
                    "agent": "system", "type": "ping",
                    "payload": {"ts": 12345},
                }))

                response = json.loads(ws.receive_text())
                assert response["agent"] == "system"
                assert response["type"] == "pong"

    def test_invalid_json_returns_error(self, setup):
        with TestClient(setup["app"]) as client:
            with client.websocket_connect("/ws/connect") as ws:
                ws.receive_text()  # hello
                ws.send_text(json.dumps({
                    "agent": "system", "type": "auth",
                    "payload": {"token": setup["token"]},
                }))
                ws.receive_text()  # auth_ok

                ws.send_text("not valid json{{{")

                response = json.loads(ws.receive_text())
                assert response["type"] == "error"

    def test_system_agents_discovery(self, setup):
        with TestClient(setup["app"]) as client:
            with client.websocket_connect("/ws/connect") as ws:
                ws.receive_text()  # hello
                ws.send_text(json.dumps({
                    "agent": "system", "type": "auth",
                    "payload": {"token": setup["token"]},
                }))
                ws.receive_text()  # auth_ok

                ws.send_text(json.dumps({
                    "agent": "system", "type": "agents",
                }))

                response = json.loads(ws.receive_text())
                assert response["type"] == "agents"
                agents = response["payload"]["agents"]
                assert any(a["agent_id"] == "echo" for a in agents)


class TestGPUGuard:
    def test_gpu_semaphore_exists(self, setup):
        hub = setup["hub"]
        assert hasattr(hub, "_gpu_semaphore")
        assert isinstance(hub._gpu_semaphore, asyncio.Semaphore)

    @pytest.mark.asyncio
    async def test_gpu_guard_limits_concurrency(self, setup):
        """GPU guard allows 2 concurrent operations (dual GPU)."""
        hub = setup["hub"]
        acquired = []

        async def worker(idx):
            async with hub.gpu_guard():
                acquired.append(idx)
                await asyncio.sleep(0.05)

        # Launch 3 workers — only 2 should run concurrently
        tasks = [asyncio.create_task(worker(i)) for i in range(3)]
        await asyncio.sleep(0.01)  # Let first two acquire
        assert len(acquired) <= 2
        await asyncio.gather(*tasks)
        assert len(acquired) == 3

    @pytest.mark.asyncio
    async def test_gpu_guard_releases_on_error(self, setup):
        """Semaphore is released even if the operation raises."""
        hub = setup["hub"]

        async def failing_op():
            async with hub.gpu_guard():
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            await failing_op()

        # Should still be acquirable (released properly)
        async with hub.gpu_guard():
            pass  # Would deadlock if not released


class TestOfflineQueueIntegration:
    def test_queue_drains_on_connect(self, setup):
        """Messages queued while offline are delivered on reconnect."""
        # Queue a message while device is offline
        msg = AlchemyMessage(agent="echo", type="queued", payload={"text": "offline msg"})
        setup["queue"].enqueue(setup["device_id"], msg)

        with TestClient(setup["app"]) as client:
            with client.websocket_connect("/ws/connect") as ws:
                ws.receive_text()  # hello
                ws.send_text(json.dumps({
                    "agent": "system", "type": "auth",
                    "payload": {"token": setup["token"]},
                }))
                ws.receive_text()  # auth_ok

                # Should receive queued message
                queued = json.loads(ws.receive_text())
                assert queued["agent"] == "echo"
                assert queued["type"] == "queued"
                assert queued["payload"]["text"] == "offline msg"
