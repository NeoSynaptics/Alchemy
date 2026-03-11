#!/usr/bin/env python3
"""The Serpentine Test — one unattended walk through every major path.

Run: python testing/serpentine.py
Output: testing/results/serpentine_{timestamp}.json
"""

import asyncio
import httpx
import json
import time
import subprocess
import sys
from datetime import datetime
from pathlib import Path

NEOSY = "http://localhost:8001"
ALCHEMY = "http://localhost:8000"
RESULTS_DIR = Path(__file__).parent / "results"


class SerpentineRunner:
    def __init__(self):
        self.steps = []
        self.failed = False

    async def step(self, name: str, fn):
        """Run a step, log timing and result."""
        print(f"  [{len(self.steps)+1:2d}] {name}...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            result = await fn()
            elapsed = time.perf_counter() - t0
            self.steps.append({
                "step": len(self.steps) + 1,
                "name": name,
                "status": "PASS",
                "elapsed_s": round(elapsed, 3),
                "detail": result,
            })
            print(f"PASS ({elapsed:.1f}s)")
            return result
        except Exception as e:
            elapsed = time.perf_counter() - t0
            self.steps.append({
                "step": len(self.steps) + 1,
                "name": name,
                "status": "FAIL",
                "elapsed_s": round(elapsed, 3),
                "error": str(e),
            })
            print(f"FAIL ({elapsed:.1f}s) — {e}")
            self.failed = True
            return None

    def save_results(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        RESULTS_DIR.mkdir(exist_ok=True)
        path = RESULTS_DIR / f"serpentine_{ts}.json"
        report = {
            "timestamp": ts,
            "total_steps": len(self.steps),
            "passed": sum(1 for s in self.steps if s["status"] == "PASS"),
            "failed": sum(1 for s in self.steps if s["status"] == "FAIL"),
            "total_time_s": round(sum(s["elapsed_s"] for s in self.steps), 1),
            "steps": self.steps,
        }
        path.write_text(json.dumps(report, indent=2))
        print(f"\n  Results saved to {path}")
        return report


async def main():
    runner = SerpentineRunner()
    memory_ids = []

    # Step 1: Health check — both services
    async def check_health():
        async with httpx.AsyncClient(timeout=10) as c:
            r_neosy = await c.get(f"{NEOSY}/health")
            assert r_neosy.status_code == 200, f"NEOSY health failed: {r_neosy.status_code}"
            neosy_data = r_neosy.json()
            assert neosy_data.get("status") == "ok", f"NEOSY not ok: {neosy_data}"

            r_alchemy = await c.get(f"{ALCHEMY}/health")
            assert r_alchemy.status_code == 200, f"Alchemy health failed: {r_alchemy.status_code}"
            alchemy_data = r_alchemy.json()
            assert alchemy_data.get("status") == "ok", f"Alchemy not ok: {alchemy_data}"

        return f"NEOSY healthy | Alchemy v{alchemy_data.get('version', '?')} healthy"

    # Step 2: Ingest 5 text files
    async def ingest_texts():
        ids = []
        async with httpx.AsyncClient(timeout=30) as c:
            for i, text in enumerate([
                "Pole vault approach: 16 steps, 8.2 m/s final speed",
                "Paella valenciana: sofrito, saffron, socarrat",
                "Fibonacci iterative: O(n) time, O(1) space",
                "Neurociencia: el cerebro consume 20% de la energia",
                "3Blue1Brown: vectors are elements of a vector space",
            ]):
                r = await c.post(f"{NEOSY}/ingest", json={"text": text, "title": f"Serpentine #{i+1}"})
                assert r.status_code == 200, f"Ingest {i} failed: {r.text}"
                ids.append(r.json()["memory_id"])
        memory_ids.extend(ids)
        return f"{len(ids)} texts ingested"

    # Step 3: Search for known content
    async def search_pole_vault():
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(f"{NEOSY}/search", json={"text": "pole vault approach speed"})
            assert r.status_code == 200
            results = r.json()
            assert len(results) > 0, "Search returned nothing!"
            return f"Found {len(results)} results"

    # Step 4: Pin a result
    async def pin_result():
        if not memory_ids:
            raise RuntimeError("No memories to pin")
        mid = memory_ids[0]
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(f"{NEOSY}/memories/{mid}/pin", json={"entity": "user", "reason": "Serpentine test"})
            assert r.status_code == 200
            return f"Pinned {mid}"

    # Step 5: Batch ingest 100 items + Alchemy APU smoke check under load
    async def batch_ingest():
        items = [{"text": f"Batch serpentine item {i}", "title": f"Batch #{i}"} for i in range(100)]
        async with httpx.AsyncClient(timeout=120) as c:
            t0 = time.perf_counter()
            r = await c.post(f"{NEOSY}/ingest/batch", json={"items": items, "entity": "user"})
            batch_elapsed = time.perf_counter() - t0
            assert r.status_code == 200
            data = r.json()
            assert data["completed"] == 100, f"Only {data['completed']}/100 completed"

            # Alchemy APU status — verify GPU orchestrator alive while NEOSY was under load
            r_apu = await c.get(f"{ALCHEMY}/v1/apu/status", timeout=10)
            assert r_apu.status_code == 200, f"APU status failed: {r_apu.status_code}"

        return (
            f"100 items ingested in {batch_elapsed:.1f}s, {data['completed']} completed"
            f" | Alchemy APU responsive"
        )

    # Step 6: Search during/after batch (latency check)
    async def search_latency():
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(f"{NEOSY}/search", json={"text": "fibonacci algorithm"})
        latency = time.perf_counter() - t0
        assert latency < 0.5, f"Search too slow: {latency:.2f}s"
        return f"Search latency: {latency*1000:.0f}ms"

    # Step 7: Restart and verify
    async def restart_and_verify():
        subprocess.run(["docker", "compose", "-f",
            "C:/Users/monic/BaratzaMemory/docker/docker-compose.yml",
            "restart"], check=True, timeout=60)
        await asyncio.sleep(15)  # Wait for services

        # Verify data survived
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(f"{NEOSY}/search", json={"text": "pole vault"})
            assert r.status_code == 200
            assert len(r.json()) > 0, "Data lost after restart!"
            return "Data survived restart"

    # Run all steps
    print("\n=== SERPENTINE TEST ===\n")
    await runner.step("Health check", check_health)
    await runner.step("Ingest 5 synthetic texts", ingest_texts)
    await runner.step("Search 'pole vault'", search_pole_vault)
    await runner.step("Pin first result", pin_result)
    await runner.step("Batch ingest 100 items", batch_ingest)
    await runner.step("Search latency check", search_latency)
    await runner.step("Restart Docker + verify persistence", restart_and_verify)

    # TODO: Add when ready:
    # await runner.step("Trigger NEO classification", classify_items)
    # await runner.step("Voice command test", voice_test)
    # await runner.step("Screenshot dashboard", screenshot_dashboard)

    report = runner.save_results()
    print(f"\n  {report['passed']}/{report['total_steps']} passed in {report['total_time_s']}s")

    if runner.failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
