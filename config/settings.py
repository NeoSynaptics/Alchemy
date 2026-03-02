"""Alchemy core configuration."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # --- Ollama ---
    ollama_host: str = "http://localhost:11434"
    ollama_cpu_model: str = "ui-tars:72b"           # CPU — GUI visuomotor agent
    ollama_keep_alive: str = "10m"

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000

    # --- Auth ---
    auth_token: str = ""
    require_auth: bool = False

    # --- Shadow Desktop (WSL2) ---
    wsl_distro: str = "Ubuntu"
    display_num: int = 99
    vnc_port: int = 5900
    novnc_port: int = 6080
    resolution: str = "1920x1080x24"

    # --- Agent ---
    agent_max_steps: int = 50
    agent_screenshot_interval: float = 1.0
    agent_timeout: float = 300.0
    agent_approval_timeout: float = 60.0
    agent_history_window: int = 8


settings = Settings()
