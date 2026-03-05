"""Alchemy Boot -- auto-start orchestrator.

Runs at Windows login. Detects GPU hardware, picks the best experience tier,
pre-loads models to the right GPUs, starts Alchemy server, then idles.

Experience tiers (auto-detected from VRAM):
  Tier 1 -- Text only:    Single GPU <14GB. Load Qwen3 14B for chat.
  Tier 2 -- Voice:        Single GPU >=14GB or dual GPU. Add Whisper + Fish Speech.
  Tier 3 -- Full stack:   Dual GPU with >=12GB each. Add AlchemyClick vision models.

Usage:
  pythonw scripts/alchemy_boot.pyw          # silent background
  python  scripts/alchemy_boot.pyw          # with console output
  python  scripts/alchemy_boot.pyw --dry-run  # detect + plan, don't start
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE = Path(__file__).resolve().parent / "alchemy_boot.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("alchemy_boot")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OLLAMA_HOST = "http://localhost:11434"
ALCHEMY_DIR = Path(__file__).resolve().parents[1]

# Model VRAM costs (approximate, in MB)
MODEL_VRAM = {
    "qwen3:14b": 9000,
    "qwen2.5vl:7b": 4400,
    # Whisper + Fish Speech are non-Ollama, managed separately
}


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------
@dataclass
class GPU:
    index: int
    name: str
    vram_total_mb: int
    vram_used_mb: int
    vram_free_mb: int


def detect_gpus() -> list[GPU]:
    """Parse nvidia-smi for GPU specs."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            log.error("nvidia-smi failed: %s", result.stderr.strip())
            return []

        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append(GPU(
                    index=int(parts[0]),
                    name=parts[1],
                    vram_total_mb=int(parts[2]),
                    vram_used_mb=int(parts[3]),
                    vram_free_mb=int(parts[4]),
                ))
        return gpus
    except FileNotFoundError:
        log.error("nvidia-smi not found -- no NVIDIA GPU?")
        return []
    except Exception:
        log.exception("GPU detection failed")
        return []


# ---------------------------------------------------------------------------
# Experience tier
# ---------------------------------------------------------------------------
@dataclass
class ExperiencePlan:
    tier: int
    tier_name: str
    gpus: list[GPU]
    gpu_assignments: dict[int, list[str]] = field(default_factory=dict)
    ollama_preload: list[str] = field(default_factory=list)
    voice_ready: bool = False
    click_ready: bool = False


def plan_experience(gpus: list[GPU]) -> ExperiencePlan:
    """Decide the best experience tier based on available GPUs."""
    if not gpus:
        log.warning("No GPUs detected -- Alchemy will run CPU-only")
        return ExperiencePlan(tier=0, tier_name="CPU only", gpus=[])

    # Sort by VRAM descending -- biggest GPU first
    gpus_sorted = sorted(gpus, key=lambda g: g.vram_total_mb, reverse=True)
    total_vram = sum(g.vram_total_mb for g in gpus_sorted)
    num_gpus = len(gpus_sorted)

    # Dual GPU with enough VRAM for full stack
    if num_gpus >= 2 and gpus_sorted[0].vram_total_mb >= 14000 and gpus_sorted[1].vram_total_mb >= 10000:
        # Tier 3: Full stack
        # Biggest GPU: voice LLM (Qwen3 14B) + vision model (Qwen2.5-VL 7B)
        # Second GPU: voice I/O (Whisper + Fish Speech) -- managed by NEO-TX
        big = gpus_sorted[0]
        small = gpus_sorted[1]

        return ExperiencePlan(
            tier=3,
            tier_name="Full Stack (voice + click agent)",
            gpus=gpus_sorted,
            gpu_assignments={
                big.index: ["qwen3:14b (9GB)", "qwen2.5vl:7b (4.4GB)"],
                small.index: ["Whisper (~1GB)", "Fish Speech (~5GB)", "swap slot (~6GB)"],
            },
            ollama_preload=["qwen3:14b", "qwen2.5vl:7b"],
            voice_ready=True,
            click_ready=True,
        )

    # Single GPU (or dual but second too small) with enough for voice
    biggest = gpus_sorted[0]
    if biggest.vram_total_mb >= 14000:
        # Tier 2: Voice (single GPU, VRAM swapping)
        return ExperiencePlan(
            tier=2,
            tier_name="Voice (single GPU, VRAM swap)",
            gpus=gpus_sorted,
            gpu_assignments={
                biggest.index: ["qwen3:14b (9GB)", "Whisper/Fish swap"],
            },
            ollama_preload=["qwen3:14b"],
            voice_ready=True,
            click_ready=False,
        )

    # Tier 1: Text only
    return ExperiencePlan(
        tier=1,
        tier_name="Text Only (chat)",
        gpus=gpus_sorted,
        gpu_assignments={
            biggest.index: ["qwen3:14b (9GB)"],
        },
        ollama_preload=["qwen3:14b"],
        voice_ready=False,
        click_ready=False,
    )


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------
def wait_for_ollama(timeout: float = 30.0) -> bool:
    """Wait for Ollama to be reachable."""
    import urllib.request
    import urllib.error

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1.0)
    return False


def preload_model(model: str, keep_alive: str = "24h") -> bool:
    """Warm-load an Ollama model into VRAM via empty generate."""
    import urllib.request
    import urllib.error

    payload = json.dumps({
        "model": model,
        "prompt": "",
        "keep_alive": keep_alive,
    }).encode()

    try:
        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            # Consume response (streaming JSON lines)
            resp.read()
            return True
    except Exception as e:
        log.error("Failed to preload %s: %s", model, e)
        return False


def get_loaded_models() -> list[str]:
    """Get currently loaded Ollama models."""
    import urllib.request

    try:
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/ps", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Alchemy server
# ---------------------------------------------------------------------------
def is_alchemy_running() -> bool:
    """Check if Alchemy server is already responding."""
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request("http://localhost:8000/health", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def start_alchemy_server() -> subprocess.Popen | None:
    """Start Alchemy FastAPI server as a background process."""
    if is_alchemy_running():
        log.info("Alchemy already running on :8000")
        return None

    log.info("Starting Alchemy server...")
    python_exe = sys.executable
    # Use uvicorn directly to avoid make/shell issues with .pyw
    proc = subprocess.Popen(
        [
            python_exe, "-m", "uvicorn",
            "alchemy.server:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--log-level", "info",
        ],
        cwd=str(ALCHEMY_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    log.info("Alchemy server starting (PID=%d)", proc.pid)

    # Wait for it to be healthy
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if is_alchemy_running():
            log.info("Alchemy server healthy on :8000")
            return proc
        if proc.poll() is not None:
            log.error("Alchemy server exited with code %d", proc.returncode)
            return None
        time.sleep(1.0)

    log.error("Alchemy server did not become healthy within 30s")
    return proc


# ---------------------------------------------------------------------------
# Windows startup shortcut
# ---------------------------------------------------------------------------
def install_startup() -> bool:
    """Create a shortcut in Windows Startup folder."""
    if sys.platform != "win32":
        log.info("Not Windows -- skip startup install")
        return False

    try:
        import winreg
        startup_dir = Path.home() / "AppData" / "Roaming" / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
        vbs_path = startup_dir / "AlchemyBoot.vbs"

        # Use VBScript to run .pyw silently (no console flash)
        script_path = Path(__file__).resolve()
        python_exe = Path(sys.executable).parent / "pythonw.exe"
        if not python_exe.exists():
            python_exe = Path(sys.executable)

        vbs_content = f'''Set WshShell = CreateObject("WScript.Shell")
WshShell.Run """{python_exe}"" ""{script_path}""", 0, False
'''
        vbs_path.write_text(vbs_content, encoding="utf-8")
        log.info("Startup shortcut installed: %s", vbs_path)
        return True
    except Exception:
        log.exception("Failed to install startup shortcut")
        return False


def uninstall_startup() -> bool:
    """Remove the startup shortcut."""
    startup_dir = Path.home() / "AppData" / "Roaming" / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    vbs_path = startup_dir / "AlchemyBoot.vbs"
    if vbs_path.exists():
        vbs_path.unlink()
        log.info("Startup shortcut removed: %s", vbs_path)
        return True
    log.info("No startup shortcut found")
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Alchemy Boot -- system orchestrator")
    parser.add_argument("--dry-run", action="store_true", help="Detect and plan only, don't start anything")
    parser.add_argument("--install-startup", action="store_true", help="Install Windows startup shortcut")
    parser.add_argument("--uninstall-startup", action="store_true", help="Remove Windows startup shortcut")
    parser.add_argument("--keep-alive", default="24h", help="Ollama keep_alive for preloaded models (default: 24h)")
    args = parser.parse_args()

    if args.install_startup:
        install_startup()
        return

    if args.uninstall_startup:
        uninstall_startup()
        return

    log.info("=" * 60)
    log.info("Alchemy Boot starting")
    log.info("=" * 60)

    # 1. Detect GPUs
    gpus = detect_gpus()
    for g in gpus:
        log.info("  GPU %d: %s (%d MB total, %d MB free)", g.index, g.name, g.vram_total_mb, g.vram_free_mb)

    # 2. Plan experience
    plan = plan_experience(gpus)
    log.info("Experience: Tier %d -- %s", plan.tier, plan.tier_name)
    for gpu_idx, models in plan.gpu_assignments.items():
        log.info("  GPU %d: %s", gpu_idx, ", ".join(models))
    log.info("  Voice ready: %s", plan.voice_ready)
    log.info("  Click ready: %s", plan.click_ready)

    if args.dry_run:
        log.info("Dry run -- stopping here")
        print(json.dumps({
            "tier": plan.tier,
            "tier_name": plan.tier_name,
            "gpu_assignments": {str(k): v for k, v in plan.gpu_assignments.items()},
            "ollama_preload": plan.ollama_preload,
            "voice_ready": plan.voice_ready,
            "click_ready": plan.click_ready,
        }, indent=2))
        return

    # 3. Wait for Ollama
    log.info("Waiting for Ollama...")
    if not wait_for_ollama(timeout=60.0):
        log.error("Ollama not reachable after 60s -- is it installed?")
        log.error("Models will not be preloaded, but Alchemy server will still start.")
    else:
        log.info("Ollama is up")

        # 4. Preload models
        loaded = get_loaded_models()
        for model in plan.ollama_preload:
            if model in loaded:
                log.info("  %s already loaded", model)
            else:
                log.info("  Preloading %s...", model)
                if preload_model(model, keep_alive=args.keep_alive):
                    log.info("  %s loaded", model)
                else:
                    log.warning("  %s FAILED to load", model)

    # 5. Start Alchemy server
    alchemy_proc = start_alchemy_server()

    log.info("=" * 60)
    log.info("Alchemy Boot complete -- Tier %d (%s)", plan.tier, plan.tier_name)
    log.info("  Alchemy: %s", "running" if is_alchemy_running() else "FAILED")
    log.info("  Models: %s", ", ".join(get_loaded_models()) or "none")
    log.info("=" * 60)

    # 6. Keep alive -- wait for Alchemy process (or idle forever if it was already running)
    if alchemy_proc:
        try:
            alchemy_proc.wait()
            log.info("Alchemy server exited (code=%s)", alchemy_proc.returncode)
        except KeyboardInterrupt:
            log.info("Boot interrupted -- shutting down")
            alchemy_proc.terminate()
    else:
        # Alchemy was already running, just exit
        log.info("Boot script done (Alchemy was already running)")


if __name__ == "__main__":
    main()
