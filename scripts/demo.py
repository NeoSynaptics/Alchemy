"""NEO-TX Phase 1 Demo — Start shadow desktop and open Firefox.

Usage:
    python scripts/demo.py

Prerequisites:
    - WSL2 with Ubuntu installed
    - Run 'make shadow-setup' first to install dependencies
"""

import subprocess
import sys
import time


def main():
    print("=== NEO-TX Shadow Desktop Demo ===")
    print()

    # Check WSL2 availability
    print("[1/3] Checking WSL2...")
    result = subprocess.run(
        ["wsl", "-d", "Ubuntu", "--", "echo", "ok"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0 or "ok" not in result.stdout:
        print("ERROR: WSL2 Ubuntu not available.")
        print("Install with: wsl --install -d Ubuntu")
        sys.exit(1)
    print("  WSL2 Ubuntu: OK")

    # Start shadow desktop
    print("[2/3] Starting shadow desktop...")
    subprocess.run(
        ["wsl", "-d", "Ubuntu", "--", "bash", "-c",
         "cd /mnt/c/Users/info/GitHub/NEO-TX && bash wsl/start_shadow.sh"],
        timeout=30,
    )

    # Wait and open browser
    print("[3/3] Opening Firefox in shadow desktop...")
    time.sleep(2)
    subprocess.run(
        ["wsl", "-d", "Ubuntu", "--", "bash", "-c",
         "DISPLAY=:99 firefox-esr https://www.google.com &"],
        timeout=10,
    )

    print()
    print("=== Shadow Desktop Running ===")
    print()
    print("  View in browser: http://localhost:6080/vnc.html?autoconnect=true")
    print()
    print("  Press Ctrl+C to stop, then run: make shadow-stop")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopping...")
