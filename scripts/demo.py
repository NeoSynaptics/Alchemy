"""Alchemy Shadow Desktop Demo — Start shadow desktop and open Firefox.

Usage:
    python scripts/demo.py

Prerequisites:
    - WSL2 with Ubuntu installed (wsl --install -d Ubuntu)
    - Run 'make shadow-setup' first to install dependencies
"""

import asyncio
import sys

from alchemy.shadow.controller import ShadowDesktopController
from alchemy.shadow.wsl import WslRunner
from config.settings import settings


async def main():
    print("=== Alchemy Shadow Desktop Demo ===")
    print()

    # Check WSL2 availability
    print("[1/4] Checking WSL2...")
    wsl = WslRunner(distro=settings.wsl_distro, display_num=settings.display_num)

    if not wsl.is_available():
        print(f"ERROR: WSL2 {settings.wsl_distro} not available.")
        print("Install with: wsl --install -d Ubuntu")
        sys.exit(1)
    print(f"  WSL2 {settings.wsl_distro}: OK")

    # Start shadow desktop
    print("[2/4] Starting shadow desktop...")
    controller = ShadowDesktopController(
        wsl=wsl,
        display_num=settings.display_num,
        vnc_port=settings.vnc_port,
        novnc_port=settings.novnc_port,
        resolution=settings.resolution,
    )
    result = await controller.start()
    print(f"  Status: {result.status.value}")

    if result.status.value != "running":
        print("ERROR: Shadow desktop failed to start.")
        sys.exit(1)

    # Health check
    print("[3/4] Verifying services...")
    health = await controller.health()
    print(f"  Xvfb:    {'OK' if health.xvfb_running else 'FAIL'}")
    print(f"  Fluxbox: {'OK' if health.fluxbox_running else 'FAIL'}")
    print(f"  x11vnc:  {'OK' if health.vnc_running else 'FAIL'}")
    print(f"  noVNC:   {'OK' if health.novnc_running else 'FAIL'}")

    # Open Firefox
    print("[4/4] Opening Firefox in shadow desktop...")
    await controller.execute("firefox https://www.google.com &")
    await asyncio.sleep(2)

    print()
    print("=== Shadow Desktop Running ===")
    print()
    print(f"  View in browser: {result.novnc_url}")
    print(f"  VNC direct:      {result.vnc_url}")
    print()
    print("  Press Ctrl+C to stop")

    # Wait for Ctrl+C
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass

    # Cleanup
    print("\nStopping shadow desktop...")
    await controller.stop()
    print("Done.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopping...")
