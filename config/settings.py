"""Alchemy core configuration."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # --- Ollama ---
    ollama_host: str = "http://localhost:11434"
    ollama_gpu_model: str = "qwen2.5-coder:14b"    # GPU — planner, reasoning
    ollama_cpu_model: str = "ui-tars:72b"           # CPU — GUI visuomotor agent
    ollama_fast_model: str = "qwen3:8b"             # GPU (swapped) — trivial chat
    ollama_keep_alive: str = "10m"

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000

    # --- Auth ---
    auth_token: str = ""
    require_auth: bool = False

    # --- Routing ---
    triviality_threshold: float = 0.7


settings = Settings()
