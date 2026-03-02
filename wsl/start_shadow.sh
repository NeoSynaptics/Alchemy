#!/bin/bash
# Start NEO-TX Shadow Desktop services in WSL2.
# Usage: bash wsl/start_shadow.sh [display_num] [vnc_port] [novnc_port] [resolution]
set -euo pipefail

DISPLAY_NUM=${1:-99}
VNC_PORT=${2:-5900}
NOVNC_PORT=${3:-6080}
RESOLUTION=${4:-1920x1080x24}
LOG_DIR="$HOME/.neotx/logs"

export DISPLAY=":${DISPLAY_NUM}"

echo "=== Starting NEO-TX Shadow Desktop ==="
echo "  Display:    :${DISPLAY_NUM}"
echo "  VNC port:   ${VNC_PORT}"
echo "  noVNC port: ${NOVNC_PORT}"
echo "  Resolution: ${RESOLUTION}"

# Kill any existing instances
echo "[1/4] Cleaning up existing processes..."
pkill -f "Xvfb :${DISPLAY_NUM}" 2>/dev/null || true
pkill -f "fluxbox" 2>/dev/null || true
pkill -f "x11vnc.*:${DISPLAY_NUM}" 2>/dev/null || true
pkill -f "novnc_proxy.*${NOVNC_PORT}" 2>/dev/null || true
sleep 1

# Start Xvfb (virtual framebuffer)
echo "[2/4] Starting Xvfb..."
Xvfb ":${DISPLAY_NUM}" -screen 0 "${RESOLUTION}" -ac > "${LOG_DIR}/xvfb.log" 2>&1 &
XVFB_PID=$!
sleep 1

if ! kill -0 "$XVFB_PID" 2>/dev/null; then
    echo "ERROR: Xvfb failed to start. Check ${LOG_DIR}/xvfb.log"
    exit 1
fi

# Start Fluxbox (window manager)
echo "[3/4] Starting Fluxbox + x11vnc..."
DISPLAY=":${DISPLAY_NUM}" fluxbox > "${LOG_DIR}/fluxbox.log" 2>&1 &
sleep 1

# Start x11vnc (VNC server)
x11vnc -display ":${DISPLAY_NUM}" -forever -shared -nopw -rfbport "${VNC_PORT}" \
    -bg -o "${LOG_DIR}/x11vnc.log" 2>/dev/null

# Start noVNC (WebSocket bridge)
echo "[4/4] Starting noVNC..."
"$HOME/noVNC/utils/novnc_proxy" --vnc "localhost:${VNC_PORT}" --listen "${NOVNC_PORT}" \
    > "${LOG_DIR}/novnc.log" 2>&1 &
sleep 1

echo ""
echo "=== Shadow Desktop Running ==="
echo "  View in browser: http://localhost:${NOVNC_PORT}/vnc.html?autoconnect=true"
echo "  VNC direct:      localhost:${VNC_PORT}"
echo "  Logs:            ${LOG_DIR}/"
echo ""
echo "  Stop with: bash wsl/stop_shadow.sh"
