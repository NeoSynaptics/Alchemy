"""Alchemy auto-starter — ensures Alchemy is running before tests.

Uses the official alchemy_boot.pyw from the Alchemy project.
If Alchemy is already running, this is a no-op.
If not, it invokes the boot orchestrator which:
  1. Detects GPUs
  2. Plans experience tier
  3. Waits for Ollama
  4. Preloads models to correct GPUs
  5. Starts the Alchemy server on :8000
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import httpx

ALCHEMY_DIR = Path(__file__).resolve().parents[3]  # .../alchemy/syntetic_test/runners -> Alchemy/
VOICE_DIR = ALCHEMY_DIR / "alchemy" / "voice"
BOOT_SCRIPT = ALCHEMY_DIR / "scripts" / "alchemy_boot.pyw"
ALCHEMY_URL = "http://localhost:8000"
VOICE_URL = "http://localhost:8100"


def is_alchemy_running() -> bool:
    try:
        r = httpx.get(f"{ALCHEMY_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def is_voice_running() -> bool:
    try:
        r = httpx.get(f"{VOICE_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def start_alchemy(log: Callable[[str, str], None]) -> bool:
    """Ensure Alchemy is running. Uses alchemy_boot.pyw if needed.

    Returns True if Alchemy is healthy after this call.
    """
    if is_alchemy_running():
        log("Alchemy already running on :8000", "pass")
        return True

    if not BOOT_SCRIPT.exists():
        log(f"Boot script not found: {BOOT_SCRIPT}", "fail")
        log("Cannot auto-start Alchemy without the boot orchestrator.", "fail")
        return False

    log("Alchemy not running. Starting via alchemy_boot.pyw...", "info")

    try:
        # Run boot script as a background process (non-blocking)
        # It handles: GPU detection, Ollama wait, model preload, server start
        proc = subprocess.Popen(
            [sys.executable, str(BOOT_SCRIPT)],
            cwd=str(ALCHEMY_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )

        # Wait for Alchemy to become healthy.
        # Boot script preloads models then spawns uvicorn as a CHILD process.
        # Boot exits after uvicorn starts, but uvicorn needs time to init.
        # So we keep polling even after boot exits.
        log("Waiting for Alchemy to start (GPU setup + model loading)...", "info")
        deadline = time.monotonic() + 180  # 3 minutes — cold boot with model loading
        boot_exited = False

        while time.monotonic() < deadline:
            if is_alchemy_running():
                log("Alchemy server is healthy on :8000", "pass")
                return True

            # Track boot process state for logging only
            if not boot_exited and proc.poll() is not None:
                boot_exited = True
                code = proc.returncode
                if code == 0:
                    log("Boot script finished. Waiting for server to become healthy...", "info")
                else:
                    log(f"Boot script exited with code {code}. Still waiting for server...", "warn")

            time.sleep(3)

        log("Alchemy did not start within 180s", "fail")
        return False

    except Exception as e:
        log(f"Failed to run boot script: {e}", "fail")
        return False


def check_contracts(log: Callable[[str, str], None]) -> dict:
    """Check module contracts via GET /v1/modules.

    Returns dict with:
      - modules: list of module dicts
      - all_satisfied: bool
      - unsatisfied: list of module IDs with broken contracts
    """
    try:
        r = httpx.get(f"{ALCHEMY_URL}/v1/modules", timeout=10)
        if r.status_code != 200:
            log(f"Module discovery returned {r.status_code}", "warn")
            return {"modules": [], "all_satisfied": False, "unsatisfied": []}

        modules = r.json()
        unsatisfied = []
        for mod in modules:
            contract = mod.get("contract") or mod.get("contract_report") or {}
            if not contract.get("satisfied", True):
                unsatisfied.append(mod["id"])

        if unsatisfied:
            log(f"Unsatisfied contracts: {', '.join(unsatisfied)}", "warn")
        else:
            log(f"All {len(modules)} module contracts satisfied", "pass")

        return {
            "modules": modules,
            "all_satisfied": len(unsatisfied) == 0,
            "unsatisfied": unsatisfied,
        }

    except Exception as e:
        log(f"Contract check failed: {e}", "fail")
        return {"modules": [], "all_satisfied": False, "unsatisfied": []}


def start_voice(log: Callable[[str, str], None]) -> bool:
    """Ensure AlchemyVoice is running on :8100.

    Returns True if Voice server is healthy after this call.
    """
    if is_voice_running():
        log("AlchemyVoice already running on :8100", "pass")
        return True

    if not VOICE_DIR.exists():
        log(f"AlchemyVoice repo not found: {VOICE_DIR}", "fail")
        return False

    log("AlchemyVoice not running. Starting...", "info")

    try:
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "alchemy.voice.server:app",
                "--host", "127.0.0.1",
                "--port", "8100",
            ],
            cwd=str(VOICE_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )

        log("Waiting for AlchemyVoice to start...", "info")
        deadline = time.monotonic() + 30

        while time.monotonic() < deadline:
            if is_voice_running():
                log("AlchemyVoice is healthy on :8100", "pass")
                return True
            if proc.poll() is not None:
                log(f"AlchemyVoice process exited (code {proc.returncode})", "fail")
                return False
            time.sleep(2)

        log("AlchemyVoice did not start within 30s", "fail")
        return False

    except Exception as e:
        log(f"Failed to start AlchemyVoice: {e}", "fail")
        return False
