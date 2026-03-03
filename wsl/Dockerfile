# Alchemy Shadow Desktop — Reproducible Ubuntu environment
# This Dockerfile documents (and can build) the exact WSL2 environment.
#
# For WSL2 usage: just run wsl/setup.sh inside Ubuntu.
# For Docker usage: docker build -t alchemy-shadow wsl/
#                   docker run -d -p 5900:5900 -p 6080:6080 alchemy-shadow

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:99
ENV HOME=/root

# System packages (matches wsl/setup.sh)
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    xvfb \
    fluxbox \
    x11vnc \
    xdotool \
    scrot \
    xclip \
    imagemagick \
    python3-xlib \
    firefox \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# noVNC (web-based VNC client)
RUN git clone --depth 1 https://github.com/novnc/noVNC.git /root/noVNC \
    && git clone --depth 1 https://github.com/novnc/websockify.git /root/noVNC/utils/websockify

# Working directories
RUN mkdir -p /root/.alchemy/screenshots \
    /root/.alchemy/logs \
    /root/.alchemy/audit

# Startup script
COPY start_shadow.sh /opt/alchemy/start_shadow.sh
RUN chmod +x /opt/alchemy/start_shadow.sh

EXPOSE 5900 6080

CMD ["bash", "/opt/alchemy/start_shadow.sh", "99", "5900", "6080", "1920x1080x24"]
