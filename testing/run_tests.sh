#!/usr/bin/env bash
# Run tests for both Alchemy and BaratzaMemory, output results to testing/results/
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

# --- BaratzaMemory Tests ---
BARATZA_ROOT="$HOME/BaratzaMemory"
if [ -d "$BARATZA_ROOT/tests" ]; then
    echo "=== Running BaratzaMemory tests ==="
    cd "$BARATZA_ROOT"

    {
        echo "# BaratzaMemory Test Results"
        echo "Run: $TIMESTAMP"
        echo ""
        PYTHONPATH=src pytest tests/ --tb=short -q 2>&1 || true
    } > "$RESULTS_DIR/baratza_latest.md"

    echo "BaratzaMemory results written to $RESULTS_DIR/baratza_latest.md"
else
    echo "BaratzaMemory repo not found at $BARATZA_ROOT — skipping"
fi

echo ""
echo "=== Done. Results in $RESULTS_DIR/ ==="
