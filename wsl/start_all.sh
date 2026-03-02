#!/bin/bash
# Quick start all shadow desktop services. Called from Windows.
# Uses nohup + setsid so processes survive after the WSL session exits.
LOG_DIR="$HOME/.alchemy/logs"
mkdir -p "$LOG_DIR"

# Clean up
pkill -f "Xvfb :99" 2>/dev/null || true
pkill -f fluxbox 2>/dev/null || true
pkill -f x11vnc 2>/dev/null || true
pkill -f novnc_proxy 2>/dev/null || true
sleep 1

# Start Xvfb (virtual framebuffer)
nohup setsid Xvfb :99 -screen 0 1920x1080x24 -ac > "$LOG_DIR/xvfb.log" 2>&1 &
sleep 1
pgrep -f "Xvfb :99" > /dev/null && echo "[OK] Xvfb" || { echo "[FAIL] Xvfb"; exit 1; }

# Start Fluxbox (window manager)
nohup setsid env DISPLAY=:99 fluxbox > "$LOG_DIR/fluxbox.log" 2>&1 &
sleep 1
pgrep -f fluxbox > /dev/null && echo "[OK] Fluxbox" || echo "[FAIL] Fluxbox"

# Start x11vnc — unset Wayland vars so it connects to Xvfb, not WSLg
nohup setsid env -u WAYLAND_DISPLAY -u XDG_SESSION_TYPE \
    x11vnc -display :99 -forever -shared -nopw -rfbport 5900 \
    -o "$LOG_DIR/x11vnc.log" > /dev/null 2>&1 &
sleep 1
pgrep -f "x11vnc.*:99" > /dev/null && echo "[OK] x11vnc" || echo "[FAIL] x11vnc"

# Start noVNC (WebSocket bridge)
nohup setsid "$HOME/noVNC/utils/novnc_proxy" --vnc localhost:5900 --listen 6080 \
    > "$LOG_DIR/novnc.log" 2>&1 &
sleep 2
pgrep -f novnc_proxy > /dev/null && echo "[OK] noVNC" || echo "[FAIL] noVNC"

echo ""
echo "=== Shadow Desktop Running ==="
echo "  Browser: http://localhost:6080/vnc.html?autoconnect=true"
echo "  VNC:     localhost:5900"
