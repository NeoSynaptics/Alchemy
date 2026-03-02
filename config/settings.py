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

    # --- Voice ---
    voice_enabled: bool = False
    wake_word: str = "hey_neo"
    whisper_model: str = "large-v3"
    whisper_device: str = "cuda"
    piper_model: str = "en_US-lessac-medium"


settings = Settings()
