#!/bin/bash
# Stop NEO-TX Shadow Desktop services.
set -euo pipefail

DISPLAY_NUM=${1:-99}

echo "=== Stopping NEO-TX Shadow Desktop ==="

pkill -f "novnc_proxy" 2>/dev/null && echo "  Stopped noVNC" || echo "  noVNC not running"
pkill -f "x11vnc.*:${DISPLAY_NUM}" 2>/dev/null && echo "  Stopped x11vnc" || echo "  x11vnc not running"
pkill -f "fluxbox" 2>/dev/null && echo "  Stopped Fluxbox" || echo "  Fluxbox not running"
pkill -f "Xvfb :${DISPLAY_NUM}" 2>/dev/null && echo "  Stopped Xvfb" || echo "  Xvfb not running"

echo ""
echo "=== Shadow Desktop Stopped ==="
