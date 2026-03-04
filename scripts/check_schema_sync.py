#!/usr/bin/env python3
"""Check that Alchemy and NEO-TX schemas.py files are in sync.

Usage:
    python scripts/check_schema_sync.py
    # Or from Makefile: make check-schemas

Exits 0 if files match, 1 if they differ.
"""

import sys
from pathlib import Path

ALCHEMY_SCHEMA = Path(__file__).resolve().parent.parent / "alchemy" / "schemas.py"
NEOTX_SCHEMA = Path(__file__).resolve().parent.parent.parent / "NEO-TX" / "neotx" / "schemas.py"


def main() -> int:
    if not ALCHEMY_SCHEMA.exists():
        print(f"ERROR: Alchemy schema not found: {ALCHEMY_SCHEMA}")
        return 1
    if not NEOTX_SCHEMA.exists():
        print(f"WARNING: NEO-TX schema not found: {NEOTX_SCHEMA}")
        print("  Skipping sync check (NEO-TX repo not found at expected path)")
        return 0

    alchemy_content = ALCHEMY_SCHEMA.read_text(encoding="utf-8")
    neotx_content = NEOTX_SCHEMA.read_text(encoding="utf-8")

    if alchemy_content == neotx_content:
        print("OK: schemas.py files are in sync")
        return 0

    print("MISMATCH: schemas.py files differ between Alchemy and NEO-TX")
    print(f"  Alchemy: {ALCHEMY_SCHEMA}")
    print(f"  NEO-TX:  {NEOTX_SCHEMA}")
    print()
    print("To fix, copy the updated file:")
    print(f'  cp "{ALCHEMY_SCHEMA}" "{NEOTX_SCHEMA}"')
    print("  (or vice versa, depending on which was updated)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
