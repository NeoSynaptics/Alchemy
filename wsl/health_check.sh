#!/bin/bash
# Check health of NEO-TX Shadow Desktop services.
set -euo pipefail

DISPLAY_NUM=${1:-99}
VNC_PORT=${2:-5900}
NOVNC_PORT=${3:-6080}
ALL_OK=true

echo "=== NEO-TX Shadow Desktop Health Check ==="

# Check Xvfb
if pgrep -f "Xvfb :${DISPLAY_NUM}" > /dev/null 2>&1; then
    echo "  [OK] Xvfb on :${DISPLAY_NUM}"
else
    echo "  [FAIL] Xvfb not running"
    ALL_OK=false
fi

# Check Fluxbox
if pgrep -f "fluxbox" > /dev/null 2>&1; then
    echo "  [OK] Fluxbox"
else
    echo "  [FAIL] Fluxbox not running"
    ALL_OK=false
fi

# Check x11vnc
if pgrep -f "x11vnc.*:${DISPLAY_NUM}" > /dev/null 2>&1; then
    echo "  [OK] x11vnc on port ${VNC_PORT}"
else
    echo "  [FAIL] x11vnc not running"
    ALL_OK=false
fi

# Check noVNC
if pgrep -f "novnc_proxy.*${NOVNC_PORT}" > /dev/null 2>&1; then
    echo "  [OK] noVNC on port ${NOVNC_PORT}"
else
    echo "  [FAIL] noVNC not running"
    ALL_OK=false
fi

# Check display works
if DISPLAY=":${DISPLAY_NUM}" xdotool getdisplaygeometry > /dev/null 2>&1; then
    GEOM=$(DISPLAY=":${DISPLAY_NUM}" xdotool getdisplaygeometry)
    echo "  [OK] Display geometry: ${GEOM}"
else
    echo "  [FAIL] Cannot query display :${DISPLAY_NUM}"
    ALL_OK=false
fi

echo ""
if $ALL_OK; then
    echo "=== All services healthy ==="
    exit 0
else
    echo "=== Some services unhealthy ==="
    exit 1
fi
