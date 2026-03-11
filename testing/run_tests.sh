#!/usr/bin/env bash
# Run tests for both Alchemy and NEOSY, output results to testing/results/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M')

mkdir -p "$RESULTS_DIR"

# --- Alchemy Tests ---
echo "=== Running Alchemy tests ==="
ALCHEMY_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ALCHEMY_ROOT"

{
    echo "# Alchemy Test Results"
    echo "Run: $TIMESTAMP"
    echo ""
    pytest tests/ --tb=short -q 2>&1 || true
} > "$RESULTS_DIR/alchemy_latest.md"

echo "Alchemy results written to $RESULTS_DIR/alchemy_latest.md"

# --- NEOSY Tests ---
NEOSY_ROOT="$HOME/BaratzaMemory"
if [ -d "$NEOSY_ROOT/tests" ]; then
    echo "=== Running NEOSY tests ==="
    cd "$NEOSY_ROOT"

    {
        echo "# NEOSY Test Results"
        echo "Run: $TIMESTAMP"
        echo ""
        PYTHONPATH=src pytest tests/ --tb=short -q 2>&1 || true
    } > "$RESULTS_DIR/neosy_latest.md"

    echo "NEOSY results written to $RESULTS_DIR/neosy_latest.md"
else
    echo "NEOSY repo not found at $NEOSY_ROOT — skipping"
fi

echo ""
echo "=== Done. Results in $RESULTS_DIR/ ==="
