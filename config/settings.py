"""Alchemy core configuration.

Settings are organized into nested groups (one per module). The flat fields
are kept for backward compatibility -- new code should use the nested form:
    settings.gate.enabled   (preferred)
    settings.gate_enabled   (legacy, still works)

Environment variables work both ways:
    GATE__ENABLED=false     (nested, via env_nested_delimiter)
    GATE_ENABLED=false      (flat, legacy)
"""

from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings


# --- Nested settings groups (one per module) ---

class OllamaSettings(BaseModel):
    """Ollama LLM backend."""
    host: str = "http://localhost:11434"
    cpu_model: str = "rashakol/UI-TARS-72B-DPO"
    fast_model: str = "hf.co/Mungert/UI-TARS-1.5-7B-GGUF:Q4_K_M"
    keep_alive: str = "10m"
    temperature: float = 0.0
    max_tokens: int = 384
    retry_attempts: int = 3
    retry_delay: float = 1.0


class ServerSettings(BaseModel):
    """FastAPI server."""
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"


class AuthSettings(BaseModel):
    """Authentication."""
    token: str = ""
    require: bool = False
    enabled: bool = False


class ScreenshotSettings(BaseModel):
    """Screenshot capture."""
    format: str = "jpeg"
    jpeg_quality: int = 85
    resize_width: int = 1280
    resize_height: int = 720


class ClickSettings(BaseModel):
    """AlchemyClick agent loop behavior."""
    max_steps: int = 50
    screenshot_interval: float = 1.0
    timeout: float = 300.0
    approval_timeout: float = 60.0
    history_window: int = 4
    use_streaming: bool = True
    model_routing: bool = True


class RouterSettings(BaseModel):
    """Context router."""
    enabled: bool = True
    detect_shadow_apps: bool = True
    detect_windows_apps: bool = True
    category_hints: bool = True
    recovery_nudges: bool = True
    completion_criteria: bool = True


class PlaywrightSettings(BaseModel):
    """Playwright agent (Tier 1)."""
    enabled: bool = True
    model: str = "qwen3:14b"
    think: bool = True
    temperature: float = 0.1
    max_tokens: int = 1024
    max_steps: int = 50
    settle_timeout: float = 5000
    headless: bool = True
    max_snapshot_elements: int = 75
    approval_enabled: bool = True


class EscalationSettings(BaseModel):
    """Playwright escalation (Tier 1.5 -- vision fallback)."""
    enabled: bool = True
    model: str = "qwen2.5vl:7b"
    temperature: float = 0.0
    max_tokens: int = 384
    parse_failures: int = 3
    repeated_actions: int = 3
    complexity_threshold: int = 60


class GUIActorSettings(BaseModel):
    """GUI-Actor (future -- Microsoft attention-based grounding)."""
    enabled: bool = False
    host: str = "http://localhost:8200"
    model: str = "microsoft/GUI-Actor-7B-Qwen2.5-VL"
    timeout: float = 30.0


class DesktopSettings(BaseModel):
    """Desktop agent (native Windows automation)."""
    enabled: bool = True
    model: str = "qwen2.5vl:7b"
    max_steps: int = 20
    temperature: float = 0.0
    max_tokens: int = 384
    screenshot_width: int = 1280
    screenshot_height: int = 720
    screenshot_quality: int = 75
    default_mode: str = "shadow"


class GateSettings(BaseModel):
    """Gate reviewer (Claude Code auto-approve)."""
    enabled: bool = True
    model: str = "qwen3:14b"
    timeout: float = 5.0


class ResearchSettings(BaseModel):
    """AlchemyBrowser research."""
    enabled: bool = True
    model: str = "qwen3:14b"
    think: bool = False
    temperature: float = 0.3
    max_tokens: int = 2048
    max_queries: int = 10
    max_pages: int = 8
    fetch_timeout: float = 15.0
    top_k: int = 5


class BrowserSettings(BaseModel):
    """AlchemyBrowser — human-facing AI search browser."""
    port: int = 8055
    google_cse_enabled: bool = True
    bing_enabled: bool = True
    ddg_enabled: bool = True
    rrf_k: int = 60
    max_results: int = 10
    scrape_chars: int = 1500
    summary_words: int = 20
    summary_model: str = "qwen3:14b"
    ai_rerank_enabled: bool = False
    memory_enabled: bool = True


class WordSettings(BaseModel):
    """AlchemyWord AI text editor."""
    enabled: bool = True
    temperature: float = 0.7
    max_tokens: int = 1024
    suggest_debounce_ms: int = 350
    annotate_debounce_ms: int = 4000


class VoiceSettings(BaseModel):
    """AlchemyVoice — voice pipeline, smart routing, conversation, tray."""
    enabled: bool = True

    # GPU model (conversational)
    gpu_model: str = "qwen3:14b"
    gpu_model_keep_alive: str = "30m"
    gpu_mode: str = "dual"  # "single" = VRAM swap, "dual" = all resident

    # STT (Whisper)
    whisper_model: str = "large-v3"
    whisper_device: str = "cuda"

    # Wake word
    wake_word: str = "hey_neo"
    voice_wake_threshold: float = 0.5

    # VAD
    voice_vad_aggressiveness: int = 2
    voice_silence_ms: int = 800

    # TTS engine selection
    tts_engine: str = "piper"  # "piper", "fish", or "kokoro"
    piper_model: str = "en_US-lessac-medium"

    # Fish Speech
    fish_speech_port: int = 8080
    fish_speech_checkpoint: str = "checkpoints/openaudio-s1-mini"
    fish_speech_decoder_path: str = ""
    fish_speech_decoder_config: str = "modded_dac_vq"
    fish_speech_host: str = "127.0.0.1"
    fish_speech_startup_timeout: float = 60.0
    fish_speech_compile: bool = False
    fish_speech_sample_rate: int = 44100
    fish_speech_temperature: float = 0.8
    fish_speech_top_p: float = 0.8
    fish_speech_repetition_penalty: float = 1.1
    fish_speech_max_new_tokens: int = 1024
    fish_speech_reference_id: str = ""
    fish_speech_chunk_length: int = 200
    fish_speech_python_exe: str = ""
    fish_speech_dir: str = ""

    # Kokoro TTS
    kokoro_host: str = "127.0.0.1"
    kokoro_port: int = 8880
    kokoro_voice: str = "af_heart"

    # System tray
    tray_enabled: bool = True
    tray_novnc_url: str = "http://localhost:6080/vnc.html?autoconnect=true&resize=scale"

    # Knowledge (NEO-RX)
    neorx_host: str = "http://localhost:8110"
    knowledge_enabled: bool = True
    knowledge_max_docs: int = 3


class FlowVSAgentSettings(BaseModel):
    """AlchemyFlowVS agent — toggle only."""
    enabled: bool = False


class ConnectSettings(BaseModel):
    """AlchemyConnect — universal tunnel/bus for external apps."""
    enabled: bool = True
    auth_timeout_seconds: float = 10.0
    ping_interval_seconds: float = 30.0
    max_connections: int = 10
    offline_queue_max: int = 200
    data_dir: str = "data/connect"


class APUSettings(BaseModel):
    """APU (Alchemy Processing Unit) — GPU fleet management."""
    vram_safety_margin_mb: int = 200  # CUDA kernel overhead buffer for pre-load check
    auto_preload: bool = False  # Auto-load models on startup and periodic reconcile


class AgentsSettings(BaseModel):
    """AlchemyAgents — internal agent orchestration. Toggle per agent."""
    enabled: bool = True
    flow_vs: FlowVSAgentSettings = FlowVSAgentSettings()


class BrainPhysicsSettings(BaseModel):
    """BrainPhysics — coarse-to-fine cognitive routing with intuitive physics."""
    enabled: bool = True
    max_iterations: int = 5          # predictive processing loop max
    error_threshold: float = 0.3     # below this → accept prediction, distill
    coarse_resolution: int = 320     # first-pass screenshot downscale (px width)
    fine_resolution: int = 1280      # refinement pass resolution
    physics_timeout: float = 5.0     # max seconds for one simulation step
    consolidation_enabled: bool = True  # distill successful patterns into memory


class BaratzaSettings(BaseModel):
    """BaratzaMemory — knowledge graph (PostgreSQL + Qdrant)."""
    enabled: bool = True
    src_path: str = "C:/Users/info/GitHub/BaratzaMemory/src"
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_user: str = "baratza"
    pg_password: str = "baratza_dev_password"
    pg_database: str = "baratza"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333


class MemorySettings(BaseModel):
    """AlchemyMemory — two-layer persistent memory + AI-native search UI."""
    enabled: bool = True

    # Screenshot capture
    screenshot_interval_active: int = 30      # seconds when user is active
    screenshot_interval_idle: int = 300       # seconds when idle
    idle_threshold_seconds: int = 60          # no input for this long = idle

    # Storage (relative paths resolved from repo root; override via ALCHEMY_MEMORY__STORAGE_PATH)
    storage_path: str = "./data/memory"
    screenshot_quality: int = 70             # JPEG quality (lower = smaller files)

    # Long-term memory (timeline)
    ltm_db: str = "timeline.db"             # relative to storage_path
    chroma_path: str = ""                    # defaults to {storage_path}/chroma if empty
    chroma_collection: str = "alchemy_timeline"

    # Short-term memory (cache)
    stm_db: str = "stm.db"                  # relative to storage_path
    cache_ttl_days: int = 4
    stm_purge_interval_seconds: int = 60

    # Models
    summarizer_model: str = "qwen2.5vl:7b"
    embedder_model: str = "nomic-embed-text"
    synthesis_model: str = "qwen3:14b"
    classifier_model: str = "qwen3:3b"
    classifier_enabled: bool = False         # disabled — concept good, needs polish (qwen3:3b 404s)
    synthesis_think: bool = True             # qwen3:14b think=true for search synthesis

    # Search
    max_ltm_results: int = 10
    max_stm_results: int = 5
    max_internet_results: int = 5

    # Phone import
    phone_import_enabled: bool = True
    vlm_worker_batch_size: int = 50
    vlm_worker_delay: float = 0.0         # seconds between VLM calls (0 = no throttle)
    vlm_auto_start: bool = True           # auto-start VLM worker after import

    @model_validator(mode="after")
    def _derive_chroma_path(self):
        if not self.chroma_path:
            self.chroma_path = f"{self.storage_path}/chroma"
        return self


# --- Root settings (composes all groups) ---

class Settings(BaseSettings):
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "env_nested_delimiter": "__",
    }

    # === Nested groups (new canonical form) ===
    ollama: OllamaSettings = OllamaSettings()
    server: ServerSettings = ServerSettings()
    auth: AuthSettings = AuthSettings()
    screenshot: ScreenshotSettings = ScreenshotSettings()
    click: ClickSettings = ClickSettings()
    router: RouterSettings = RouterSettings()
    pw: PlaywrightSettings = PlaywrightSettings()
    pw_escalation: EscalationSettings = EscalationSettings()
    gui_actor: GUIActorSettings = GUIActorSettings()
    desktop: DesktopSettings = DesktopSettings()
    gate: GateSettings = GateSettings()
    research: ResearchSettings = ResearchSettings()
    word: WordSettings = WordSettings()
    voice: VoiceSettings = VoiceSettings()
    connect: ConnectSettings = ConnectSettings()
    apu: APUSettings = APUSettings()
    agents: AgentsSettings = AgentsSettings()
    memory: MemorySettings = MemorySettings()
    browser: BrowserSettings = BrowserSettings()
    brain_physics: BrainPhysicsSettings = BrainPhysicsSettings()
    baratza: BaratzaSettings = BaratzaSettings()

    # === Flat fields (backward compat -- used by server.py and existing code) ===

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_cpu_model: str = "rashakol/UI-TARS-72B-DPO"
    ollama_fast_model: str = "hf.co/Mungert/UI-TARS-1.5-7B-GGUF:Q4_K_M"
    ollama_keep_alive: str = "10m"
    ollama_temperature: float = 0.0
    ollama_max_tokens: int = 384
    ollama_retry_attempts: int = 3
    ollama_retry_delay: float = 1.0

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # Auth
    auth_token: str = ""
    require_auth: bool = False
    security_enabled: bool = False

    # Screenshot
    screenshot_format: str = "jpeg"
    screenshot_jpeg_quality: int = 85
    screenshot_resize_width: int = 1280
    screenshot_resize_height: int = 720

    # AlchemyClick (legacy: agent_*)
    click_max_steps: int = 50
    click_screenshot_interval: float = 1.0
    click_timeout: float = 300.0
    click_approval_timeout: float = 60.0
    click_history_window: int = 4
    click_use_streaming: bool = True
    click_model_routing: bool = True
    click_omniparser_enabled: bool = False
    click_omniparser_confidence: float = 0.3
    click_omniparser_device: str = "cuda:0"
    click_omniparser_model_path: str = ""

    # Router
    router_enabled: bool = True
    router_detect_shadow_apps: bool = True
    router_detect_windows_apps: bool = True
    router_category_hints: bool = True
    router_recovery_nudges: bool = True
    router_completion_criteria: bool = True

    # Playwright Agent (Tier 1)
    pw_enabled: bool = True
    pw_model: str = "qwen3:14b"
    pw_think: bool = True
    pw_temperature: float = 0.1
    pw_max_tokens: int = 1024
    pw_max_steps: int = 50
    pw_settle_timeout: float = 5000
    pw_headless: bool = True
    pw_max_snapshot_elements: int = 75
    pw_approval_enabled: bool = True

    # Playwright Escalation (Tier 1.5)
    pw_escalation_enabled: bool = True
    pw_escalation_model: str = "qwen2.5vl:7b"
    pw_escalation_temperature: float = 0.0
    pw_escalation_max_tokens: int = 384
    pw_escalation_parse_failures: int = 3
    pw_escalation_repeated_actions: int = 3
    pw_escalation_complexity_threshold: int = 60

    # GUI-Actor (future)
    gui_actor_enabled: bool = False
    gui_actor_host: str = "http://localhost:8200"
    gui_actor_model: str = "microsoft/GUI-Actor-7B-Qwen2.5-VL"
    gui_actor_timeout: float = 30.0

    # Desktop Agent
    desktop_enabled: bool = True
    desktop_model: str = "qwen2.5vl:7b"
    desktop_max_steps: int = 20
    desktop_temperature: float = 0.0
    desktop_max_tokens: int = 384
    desktop_screenshot_width: int = 1280
    desktop_screenshot_height: int = 720
    desktop_screenshot_quality: int = 75
    desktop_default_mode: str = "shadow"

    # Gate
    gate_enabled: bool = True
    gate_model: str = "qwen3:14b"
    gate_timeout: float = 5.0

    # Voice (flat compat — new code should use settings.voice.*)
    voice_enabled: bool = True
    tts_engine: str = "piper"
    gpu_model: str = "qwen3:14b"
    gpu_model_keep_alive: str = "30m"
    gpu_mode: str = "dual"
    whisper_model: str = "large-v3"
    whisper_device: str = "cuda"
    wake_word: str = "hey_neo"
    voice_wake_threshold: float = 0.5
    voice_vad_aggressiveness: int = 2
    voice_silence_ms: int = 800
    piper_model: str = "en_US-lessac-medium"
    fish_speech_port: int = 8080
    fish_speech_checkpoint: str = "checkpoints/openaudio-s1-mini"
    fish_speech_decoder_path: str = ""
    fish_speech_decoder_config: str = "modded_dac_vq"
    fish_speech_host: str = "127.0.0.1"
    fish_speech_startup_timeout: float = 60.0
    fish_speech_compile: bool = False
    fish_speech_sample_rate: int = 44100
    fish_speech_temperature: float = 0.8
    fish_speech_top_p: float = 0.8
    fish_speech_repetition_penalty: float = 1.1
    fish_speech_max_new_tokens: int = 1024
    fish_speech_reference_id: str = ""
    fish_speech_chunk_length: int = 200
    fish_speech_python_exe: str = ""
    fish_speech_dir: str = ""
    kokoro_host: str = "127.0.0.1"
    kokoro_port: int = 8880
    kokoro_voice: str = "af_heart"
    tray_enabled: bool = True
    tray_novnc_url: str = "http://localhost:6080/vnc.html?autoconnect=true&resize=scale"
    neorx_host: str = "http://localhost:8110"
    knowledge_enabled: bool = True
    knowledge_max_docs: int = 3

    # AlchemyConnect
    connect_enabled: bool = True

    # BrainPhysics
    brain_physics_enabled: bool = True

    # Research
    research_enabled: bool = True
    research_model: str = "qwen3:14b"
    research_think: bool = False
    research_temperature: float = 0.3
    research_max_tokens: int = 2048
    research_max_queries: int = 10
    research_max_pages: int = 8
    research_fetch_timeout: float = 15.0
    research_top_k: int = 5


settings = Settings()
