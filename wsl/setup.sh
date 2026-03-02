#!/bin/bash
# NEO-TX WSL2 Shadow Desktop Setup
# Run this inside WSL2 Ubuntu to install all shadow desktop dependencies.
set -euo pipefail

echo "=== NEO-TX Shadow Desktop Setup ==="

# System packages
echo "[1/4] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    xvfb \
    fluxbox \
    x11vnc \
    xdotool \
    scrot \
    xclip \
    imagemagick \
    python3-xlib \
    firefox-esr 2>/dev/null

# noVNC (web-based VNC client)
echo "[2/4] Installing noVNC..."
NOVNC_DIR="$HOME/noVNC"
if [ ! -d "$NOVNC_DIR" ]; then
    git clone --depth 1 https://github.com/novnc/noVNC.git "$NOVNC_DIR"
    git clone --depth 1 https://github.com/novnc/websockify.git "$NOVNC_DIR/utils/websockify"
else
    echo "  noVNC already installed at $NOVNC_DIR"
fi

# Create working directories
echo "[3/4] Creating working directories..."
mkdir -p "$HOME/.neotx/screenshots"
mkdir -p "$HOME/.neotx/logs"
mkdir -p "$HOME/.neotx/audit"

# Verify installations
echo "[4/4] Verifying installations..."
echo "  Xvfb:    $(which Xvfb)"
echo "  fluxbox: $(which fluxbox)"
echo "  x11vnc:  $(which x11vnc)"
echo "  xdotool: $(which xdotool)"
echo "  scrot:   $(which scrot)"
echo "  noVNC:   $NOVNC_DIR/utils/novnc_proxy"

echo ""
echo "=== Setup complete ==="
echo "Run 'bash wsl/start_shadow.sh' to start the shadow desktop."
