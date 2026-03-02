"""Integration test — full Alchemy ↔ NEO-TX handshake.

Starts both servers on test ports, verifies the complete task lifecycle:
health check → submit task → poll status → approval flow → shadow control.

Run with: pytest tests/test_integration.py -m integration -v
"""

import subprocess
import sys
import time
from uuid import UUID

import httpx
import pytest

ALCHEMY_PORT = 18000
NEOTX_PORT = 18100
ALCHEMY_URL = f"http://127.0.0.1:{ALCHEMY_PORT}"
NEOTX_URL = f"http://127.0.0.1:{NEOTX_PORT}"


def _wait_for_server(url: str, timeout: float = 10.0):
    """Poll until server responds to /health."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{url}/health", timeout=2.0)
            if r.status_code == 200:
                return
        except httpx.ConnectError:
            pass
        time.sleep(0.3)
    raise TimeoutError(f"Server at {url} did not start within {timeout}s")


@pytest.fixture(scope="module")
def servers():
    """Start both servers on test ports, yield URLs, then kill them."""
    alchemy = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "alchemy.server:app",
         "--host", "127.0.0.1", "--port", str(ALCHEMY_PORT)],
        cwd="C:/Users/info/GitHub/Alchemy",
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    neotx = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "neotx.server:app",
         "--host", "127.0.0.1", "--port", str(NEOTX_PORT)],
        cwd="C:/Users/info/GitHub/NEO-TX",
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    try:
        _wait_for_server(ALCHEMY_URL)
        _wait_for_server(NEOTX_URL)
        yield {"alchemy": ALCHEMY_URL, "neotx": NEOTX_URL}
    finally:
        alchemy.terminate()
        neotx.terminate()
        alchemy.wait(timeout=5)
        neotx.wait(timeout=5)


@pytest.mark.integration
class TestFullHandshake:
    def test_health_checks(self, servers):
        r1 = httpx.get(f"{servers['alchemy']}/health")
        r2 = httpx.get(f"{servers['neotx']}/health")
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json()["status"] == "ok"
        assert r2.json()["status"] == "ok"

    def test_submit_task_and_poll(self, servers):
        r = httpx.post(f"{servers['alchemy']}/vision/task", json={
            "goal": "send email with hours",
            "callback_url": servers["neotx"],
        })
        assert r.status_code == 200
        data = r.json()
        task_id = data["task_id"]
        assert UUID(task_id)
        assert data["status"] == "pending"

        r2 = httpx.get(f"{servers['alchemy']}/vision/task/{task_id}/status")
        assert r2.status_code == 200
        assert r2.json()["task_id"] == task_id

    def test_approval_flow(self, servers):
        # Submit task
        r = httpx.post(f"{servers['alchemy']}/vision/task", json={
            "goal": "test approval", "callback_url": servers["neotx"],
        })
        task_id = r.json()["task_id"]

        # Alchemy sends approval callback to NEO-TX
        r2 = httpx.post(f"{servers['neotx']}/callbacks/approval", json={
            "task_id": task_id,
            "action": {"action": "click", "x": 340, "y": 200,
                       "reasoning": "Click Send", "tier": "approve"},
            "screenshot_b64": "iVBORw0KGgo=",
            "step": 3, "timeout_seconds": 60, "goal": "test approval",
        })
        assert r2.status_code == 200
        assert r2.json()["received"] is True

        # NEO-TX approves
        r3 = httpx.post(f"{servers['alchemy']}/vision/task/{task_id}/approve", json={
            "decided_by": "user", "reason": "looks good",
        })
        assert r3.status_code == 200
        assert r3.json()["decision"] == "approved"
        assert r3.json()["status"] == "running"

    def test_deny_flow(self, servers):
        r = httpx.post(f"{servers['alchemy']}/vision/task", json={
            "goal": "test denial", "callback_url": servers["neotx"],
        })
        task_id = r.json()["task_id"]

        r2 = httpx.post(f"{servers['alchemy']}/vision/task/{task_id}/deny", json={
            "decided_by": "user", "reason": "wrong recipient",
        })
        assert r2.status_code == 200
        assert r2.json()["decision"] == "denied"

    def test_vision_analyze(self, servers):
        r = httpx.post(f"{servers['alchemy']}/vision/analyze", json={
            "screenshot_b64": "iVBORw0KGgo=",
            "goal": "find the search box",
        })
        assert r.status_code == 200
        assert r.json()["action"]["action"] in ("click", "type", "scroll", "done", "fail")

    def test_shadow_lifecycle(self, servers):
        r1 = httpx.post(f"{servers['alchemy']}/shadow/start", json={})
        assert r1.status_code == 200
        assert r1.json()["status"] == "running"

        r2 = httpx.get(f"{servers['alchemy']}/shadow/health")
        assert r2.status_code == 200

        r3 = httpx.post(f"{servers['alchemy']}/shadow/stop")
        assert r3.status_code == 200
        assert r3.json()["status"] == "stopped"

    def test_screenshot_returns_png(self, servers):
        r = httpx.get(f"{servers['alchemy']}/shadow/screenshot")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
        assert r.content[:4] == b"\x89PNG"

    def test_models_endpoint(self, servers):
        r = httpx.get(f"{servers['alchemy']}/models")
        assert r.status_code == 200
        assert len(r.json()["models"]) >= 1

    def test_notify_callback(self, servers):
        r = httpx.post(f"{servers['neotx']}/callbacks/notify", json={
            "task_id": "00000000-0000-0000-0000-000000000001",
            "action": {"action": "click", "x": 50, "y": 50,
                       "reasoning": "Opening Firefox", "tier": "notify"},
            "message": "Opening Firefox", "step": 1,
        })
        assert r.status_code == 200
        assert r.json()["received"] is True

    def test_task_update_callback(self, servers):
        r = httpx.post(f"{servers['neotx']}/callbacks/task-update", json={
            "task_id": "00000000-0000-0000-0000-000000000001",
            "status": "completed", "current_step": 5,
            "message": "Task done",
        })
        assert r.status_code == 200
        assert r.json()["received"] is True
