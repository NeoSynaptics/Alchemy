"""Alchemy core configuration."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # --- Ollama ---
    ollama_host: str = "http://localhost:11434"
    ollama_cpu_model: str = "rashakol/UI-TARS-72B-DPO"  # CPU — GUI visuomotor agent
    ollama_fast_model: str = "hf.co/Mungert/UI-TARS-1.5-7B-GGUF:Q4_K_M"  # Fast dev/simple tasks
    ollama_keep_alive: str = "10m"
    ollama_temperature: float = 0.0  # Deterministic for GUI actions
    ollama_max_tokens: int = 384  # Actions are short — save context budget
    ollama_retry_attempts: int = 3
    ollama_retry_delay: float = 1.0  # Seconds between retries (exponential)

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # --- Auth ---
    auth_token: str = ""
    require_auth: bool = False

    # --- Shadow Desktop (WSL2) ---
    wsl_distro: str = "Ubuntu"
    shadow_wsl_repo_path: str = "/mnt/c/Users/info/GitHub/Alchemy"
    display_num: int = 99
    vnc_port: int = 5900
    novnc_port: int = 6080
    resolution: str = "1920x1080x24"

    # --- Screenshot ---
    screenshot_format: str = "jpeg"  # jpeg or png — jpeg is ~5x smaller
    screenshot_jpeg_quality: int = 85  # JPEG quality (0-100)
    screenshot_resize_width: int = 1280  # Downscale to reduce visual tokens
    screenshot_resize_height: int = 720  # 720p = ~40% fewer tokens than 1080p

    # --- Agent ---
    agent_max_steps: int = 50
    agent_screenshot_interval: float = 1.0
    agent_timeout: float = 300.0
    agent_approval_timeout: float = 60.0
    agent_history_window: int = 4  # Text-only for older steps saves tokens
    agent_use_streaming: bool = True  # Stream inference, stop on Action:
    agent_model_routing: bool = True  # Use fast model for simple tasks

    # --- Router (context injection) ---
    router_enabled: bool = True
    router_detect_shadow_apps: bool = True
    router_detect_windows_apps: bool = True
    router_category_hints: bool = True
    router_recovery_nudges: bool = True
    router_completion_criteria: bool = True

    # --- Playwright Agent (Tier 1) ---
    pw_enabled: bool = True
    pw_model: str = "qwen3:14b"
    pw_think: bool = True  # Qwen3 needs think=true to follow agent instructions
    pw_temperature: float = 0.1  # Low temp for deterministic actions
    pw_max_tokens: int = 1024  # Room for thinking + Thought: + Action: output
    pw_max_steps: int = 50
    pw_settle_timeout: float = 5000  # ms to wait for page settle after action
    pw_headless: bool = True  # Run Chromium headless
    pw_max_snapshot_elements: int = 75  # Max refs — 150 overwhelms 14B models
    pw_approval_enabled: bool = True  # Pause on irreversible actions

    # --- Playwright Escalation (Tier 1.5 — vision fallback) ---
    pw_escalation_enabled: bool = True  # Enable UI-TARS 7B fallback when stuck
    pw_escalation_model: str = "minicpm-v"  # Vision model — pixel coordinate output
    pw_escalation_temperature: float = 0.0
    pw_escalation_max_tokens: int = 384
    pw_escalation_parse_failures: int = 3  # Consecutive parse errors before escalating
    pw_escalation_repeated_actions: int = 3  # Same action N times = loop
    pw_escalation_complexity_threshold: int = 60  # Ref count that triggers escalation


    # --- Gate (Claude Code auto-approve) ---
    gate_enabled: bool = True
    gate_model: str = "qwen3:14b"  # Same model, think:false mode for speed
    gate_timeout: float = 5.0  # Max inference time (seconds)

    # --- Research (AlchemyBrowser) ---
    research_enabled: bool = True
    research_model: str = "qwen3:14b"
    research_think: bool = False  # think:false for speed — decomposition + synthesis
    research_temperature: float = 0.3  # Slightly creative for synthesis, still focused
    research_max_tokens: int = 2048  # Room for long synthesized answers
    research_max_queries: int = 10  # Max sub-queries from decomposition
    research_max_pages: int = 8  # Max pages to fetch in parallel
    research_fetch_timeout: float = 15.0  # Seconds per page fetch
    research_top_k: int = 5  # Top K pages after relevance scoring


settings = Settings()
