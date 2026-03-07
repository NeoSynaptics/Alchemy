"""Create a desktop shortcut for AlchemySynteticTest."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def create_shortcut():
    desktop = Path("C:/Desktop")
    app_dir = Path(__file__).parent.resolve()
    app_py = app_dir / "app.py"
    icon = app_dir / "assets" / "alchemytest.ico"
    shortcut = desktop / "Alchemy Syntetic Test.lnk"

    python_exe = sys.executable

    # Use PowerShell to create .lnk
    ps_script = f'''
$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut("{shortcut}")
$sc.TargetPath = "{python_exe}"
$sc.Arguments = '"{app_py}"'
$sc.WorkingDirectory = "{app_dir}"
$sc.IconLocation = "{icon},0"
$sc.Description = "Alchemy Syntetic Test — E2E pipeline validation"
$sc.Save()
'''
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_script],
        check=True,
        capture_output=True,
    )
    print(f"Shortcut created: {shortcut}")


if __name__ == "__main__":
    create_shortcut()
