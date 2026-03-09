"""AlchemyMemory Desktop App — opens Memory timeline in a native window."""

import subprocess
import sys
import os
import shutil
import time
import urllib.request

UI_URL = "http://localhost:5173/memory"
UI_HEALTH = "http://localhost:5173/"
API_HEALTH = "http://localhost:8000/health"
NPM = r"C:\Program Files\nodejs\npm.cmd"

# Browser paths (prefer Edge for PWA-like experience on Windows)
BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]


def is_up(url: str) -> bool:
    try:
        urllib.request.urlopen(url, timeout=2)
        return True
    except Exception:
        return False


def find_browser() -> str | None:
    for path in BROWSERS:
        if os.path.exists(path):
            return path
    return None


def main():
    browser = find_browser()
    if not browser:
        import webbrowser
        webbrowser.open(UI_URL)
        return

    alchemy_dir = os.path.join(os.path.dirname(__file__), "..")

    # Start Alchemy backend if not running
    if not is_up(API_HEALTH):
        try:
            subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "alchemy.server:app",
                 "--host", "0.0.0.0", "--port", "8000"],
                cwd=alchemy_dir,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            for _ in range(20):
                time.sleep(1)
                if is_up(API_HEALTH):
                    break
        except Exception:
            pass

    # Start Vite dev server if not running
    if not is_up(UI_HEALTH):
        npm_cmd = NPM if os.path.exists(NPM) else shutil.which("npm") or "npm"
        try:
            subprocess.Popen(
                [npm_cmd, "run", "dev"],
                cwd=os.path.join(alchemy_dir, "ui"),
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            for _ in range(15):
                time.sleep(1)
                if is_up(UI_HEALTH):
                    break
        except Exception:
            pass

    # Open in app mode (no browser chrome, clean native window)
    subprocess.Popen([
        browser,
        f"--app={UI_URL}",
        "--window-size=1400,900",
        "--window-position=center",
        "--disable-extensions",
        "--new-window",
    ])


if __name__ == "__main__":
    main()
