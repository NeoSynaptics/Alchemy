"""Alchemy FastAPI server — model routing + management API on port 8000."""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from uuid import uuid4

# Windows: Playwright requires ProactorEventLoop for subprocess support.
# Must be set before any event loop is created (uvicorn --reload spawns a child
# that imports this module, so module-level is the right place).
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from importlib.metadata import version as _pkg_version

from config.logging import setup_logging
from fastapi import FastAPI

try:
    __version__ = _pkg_version("alchemy")
except Exception:
    __version__ = "0.4.0"

setup_logging()
from fastapi.middleware.cors import CORSMiddleware

from alchemy.adapters import OllamaClient
from alchemy.click.task_manager import TaskManager
from alchemy.api import models_api, vision
from alchemy.api import playwright_api
from alchemy.api import research_api
from alchemy.api import gate_api
from alchemy.api import desktop_api
from alchemy.api import apu_api
from alchemy.api import modules_api
from alchemy.api import click_api
from alchemy.router.environment import EnvironmentDetector
from config.settings import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    # Initialize Ollama client
    ollama = OllamaClient(
        host=settings.ollama.host,
        timeout=120.0,
        keep_alive=settings.ollama.keep_alive,
        retry_attempts=settings.ollama.retry_attempts,
        retry_delay=settings.ollama.retry_delay,
    )
    await ollama.start()

    if await ollama.ping():
        logger.info("Ollama at %s — connected", settings.ollama.host)
        # Check both models
        for model_name in [settings.ollama.cpu_model, settings.ollama.fast_model]:
            if await ollama.is_model_available(model_name):
                logger.info("Model %s — available", model_name)
            else:
                logger.warning("Model %s not found — pull with: ollama pull %s",
                              model_name, model_name)
    else:
        logger.warning("Ollama not reachable at %s", settings.ollama.host)

    app.state.ollama_client = ollama
    app.state.task_manager = TaskManager()

    # --- APU (Alchemy Processing Unit) ---
    try:
        from pathlib import Path
        from alchemy.apu import GPUMonitor, ModelRegistry, StackOrchestrator

        gpu_monitor = GPUMonitor()
        model_registry = ModelRegistry()
        fleet_path = Path(__file__).parent.parent / "config" / "gpu_fleet.yaml"
        if fleet_path.exists():
            model_registry.load_fleet_config(fleet_path)

        orchestrator = StackOrchestrator(
            monitor=gpu_monitor,
            registry=model_registry,
            ollama_host=settings.ollama_host,
        )
        await orchestrator.start()
        app.state.orchestrator = orchestrator
        app.state.gpu_monitor = gpu_monitor
        app.state.model_registry = model_registry
        logger.info("APU started (%d models registered)",
                     len(model_registry.all_models()))

        # Validate model contracts — apps declare what they need, core checks availability
        from alchemy.contracts import validate_contracts
        reports = validate_contracts(model_registry)
        app.state.contract_reports = {}
        for report in reports:
            app.state.contract_reports[report.module_id] = report
            if not report.satisfied:
                logger.warning("Module %s missing models: %s", report.module_id, report.missing)
            elif report.optional_missing:
                logger.info("Module %s: optional models unavailable: %s",
                           report.module_id, report.optional_missing)

    except Exception as e:
        logger.warning("APU failed to start: %s", e)
        app.state.orchestrator = None
        app.state.gpu_monitor = None
        app.state.model_registry = None
        app.state.contract_reports = {}

    # --- Playwright Agent (Tier 1) ---
    if settings.pw.enabled:
        try:
            from alchemy.core import PlaywrightAgent, ApprovalGate, BrowserManager

            browser_mgr = BrowserManager(headless=settings.pw.headless)
            await browser_mgr.start()
            app.state.browser_manager = browser_mgr

            approval_gate = ApprovalGate(enabled=settings.pw.approval_enabled)

            # Tier 1.5: Vision escalation (UI-TARS 7B fallback when stuck)
            vision_escalation = None
            stuck_detector = None
            if settings.pw_escalation.enabled:
                from alchemy.core import StuckDetector, VisionEscalation

                vision_escalation = VisionEscalation(
                    ollama_client=ollama,
                    model=settings.pw_escalation.model,
                    temperature=settings.pw_escalation.temperature,
                    max_tokens=settings.pw_escalation.max_tokens,
                )
                stuck_detector = StuckDetector(
                    max_parse_failures=settings.pw_escalation.parse_failures,
                    max_repeated_actions=settings.pw_escalation.repeated_actions,
                    complexity_threshold=settings.pw_escalation.complexity_threshold,
                )
                logger.info("Tier 1.5 escalation ready (model=%s)", settings.pw_escalation.model)

            pw_agent = PlaywrightAgent(
                ollama_client=ollama,
                model=settings.pw.model,
                max_steps=settings.pw.max_steps,
                think=settings.pw.think,
                temperature=settings.pw.temperature,
                max_tokens=settings.pw.max_tokens,
                settle_timeout=settings.pw.settle_timeout,
                approval_checker=lambda action: approval_gate.needs_approval(action),
                vision_escalation=vision_escalation,
                stuck_detector=stuck_detector,
            )
            app.state.pw_agent = pw_agent
            logger.info("Playwright agent ready (model=%s, think=%s, escalation=%s)",
                        settings.pw.model, settings.pw.think, settings.pw_escalation.enabled)

        except ImportError as e:
            logger.warning("Playwright not installed — Tier 1 agent disabled: %s", e)
            app.state.browser_manager = None
            app.state.pw_agent = None
        except Exception as e:
            logger.warning("Playwright agent failed to start: %s", e)
            app.state.browser_manager = None
            app.state.pw_agent = None
    else:
        app.state.browser_manager = None
        app.state.pw_agent = None

    # --- Desktop Agent (native Windows automation) ---
    if settings.desktop.enabled:
        try:
            from alchemy.desktop import DesktopAgent, DesktopController

            desktop_ctrl = DesktopController(
                screenshot_width=settings.desktop.screenshot_width,
                screenshot_height=settings.desktop.screenshot_height,
                screenshot_quality=settings.desktop.screenshot_quality,
                mode=settings.desktop.default_mode,
            )
            app.state.desktop_agent = DesktopAgent(
                ollama_client=ollama,
                controller=desktop_ctrl,
                model=settings.desktop.model,
                max_steps=settings.desktop.max_steps,
                temperature=settings.desktop.temperature,
                max_tokens=settings.desktop.max_tokens,
            )
            logger.info("Desktop agent ready (model=%s)", settings.desktop.model)
        except ImportError as e:
            logger.warning("pyautogui not installed — desktop agent disabled: %s", e)
            app.state.desktop_agent = None
        except Exception as e:
            logger.warning("Desktop agent failed to start: %s", e)
            app.state.desktop_agent = None
    else:
        app.state.desktop_agent = None

    # --- GUI-Actor (future — Microsoft attention-based grounding) ---
    app.state.gui_actor_client = None
    if settings.gui_actor.enabled:
        try:
            from alchemy.adapters import GUIActorClient

            gui_actor = GUIActorClient(
                host=settings.gui_actor.host,
                timeout=settings.gui_actor.timeout,
            )
            await gui_actor.start()
            if await gui_actor.ping():
                app.state.gui_actor_client = gui_actor
                logger.info("GUI-Actor client ready (host=%s)", settings.gui_actor.host)
            else:
                logger.warning("GUI-Actor server not reachable at %s", settings.gui_actor.host)
                await gui_actor.close()
        except Exception as e:
            logger.warning("GUI-Actor client failed to start: %s", e)

    # --- Gate (Claude Code auto-approve) ---
    if settings.gate.enabled:
        from alchemy.gate import GateReviewer

        app.state.gate_reviewer = GateReviewer(
            ollama_client=ollama,
            model=settings.gate.model,
            timeout=settings.gate.timeout,
        )
        logger.info("Gate reviewer ready (model=%s)", settings.gate.model)
    else:
        app.state.gate_reviewer = None

    # Detect environment for context router
    if settings.router.enabled:
        detector = EnvironmentDetector()
        app.state.environment = await detector.detect()
        logger.info(
            "Router environment: %d windows apps",
            len(app.state.environment.windows_apps),
        )
    else:
        app.state.environment = None

    # --- AlchemyVoice (voice pipeline, smart router, conversation, tray) ---
    app.state.voice_system = None
    if settings.voice.enabled:
        try:
            from alchemy.voice import VoiceSystem

            voice_system = VoiceSystem(settings)
            # Voice router needs the SmartRouter — build it
            from alchemy.voice.models.provider import OllamaProvider
            from alchemy.voice.models.registry import build_default_registry
            from alchemy.voice.models.schemas import ModelLocation
            from alchemy.voice.router.cascade import ConversationToVisionCascade
            from alchemy.voice.router.router import SmartRouter
            from alchemy.voice.models.conversation import ConversationManager

            voice_registry = build_default_registry()
            voice_ollama = OllamaProvider(
                host=settings.ollama.host,
                timeout=120.0,
                keep_alive=settings.voice.gpu_model_keep_alive,
            )
            await voice_ollama.start()

            voice_providers = {ModelLocation.GPU_LOCAL: voice_ollama}
            voice_router = SmartRouter(
                registry=voice_registry,
                providers=voice_providers,
                conversation_manager=ConversationManager(),
                cascades=[ConversationToVisionCascade()],
            )

            await voice_system.start(router=voice_router)
            app.state.voice_system = voice_system
            app.state.voice_ollama = voice_ollama
            logger.info("AlchemyVoice started")
        except ImportError as e:
            logger.warning("Voice dependencies not available: %s", e)
        except Exception:
            logger.exception("AlchemyVoice failed to start")

    # --- AlchemyConnect (phone-to-PC tunnel) ---
    app.state.connect = None
    if settings.connect.enabled:
        try:
            from alchemy.connect import AlchemyConnect
            from alchemy.connect.agents.chat_agent import ChatAgent
            from alchemy.connect.agents.image_agent import ImageAgent
            from alchemy.connect.agents.voice_agent import VoiceAgent

            connect = AlchemyConnect(app, settings)
            connect.register_agent(ChatAgent(app.state))
            connect.register_agent(ImageAgent(app.state, gpu_guard=connect._hub._gpu_semaphore))
            connect.register_agent(VoiceAgent(app.state))
            await connect.start()
            app.state.connect = connect
            logger.info("AlchemyConnect started (%d agents)",
                        len(connect.available_agents))
        except Exception:
            logger.exception("AlchemyConnect failed to start")

    # --- Constitution (approval defense) ---
    app.state.constitution = None
    try:
        from alchemy.voice.constitution.engine import ConstitutionEngine

        app.state.constitution = ConstitutionEngine()
        logger.info("Constitution engine loaded (%d rules)", len(app.state.constitution.rules))
    except Exception:
        logger.debug("Constitution engine not available")

    # --- Task planner ---
    app.state.planner = None
    try:
        from alchemy.voice.planner.planner import TaskPlanner

        app.state.planner = TaskPlanner()
        logger.info("Task planner initialized")
    except Exception:
        logger.debug("Task planner not available")

    yield

    # Cleanup
    if getattr(app.state, "connect", None):
        await app.state.connect.stop()
        logger.info("AlchemyConnect stopped")

    if getattr(app.state, "voice_system", None):
        await app.state.voice_system.stop()
        logger.info("AlchemyVoice stopped")
    if getattr(app.state, "voice_ollama", None):
        await app.state.voice_ollama.close()

    if getattr(app.state, "orchestrator", None):
        await app.state.orchestrator.close()
        logger.info("APU stopped")

    if getattr(app.state, "browser_manager", None):
        await app.state.browser_manager.close()
        logger.info("Playwright browser closed")

    await ollama.close()
    if getattr(app.state, "gui_actor_client", None):
        await app.state.gui_actor_client.close()


app = FastAPI(
    title="Alchemy",
    description="Local-first LLM core engine",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.auth.cors_origins,
    allow_credentials=len(settings.auth.cors_origins) > 0 and settings.auth.cors_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def auth_middleware(request, call_next):
    """Enforce bearer token auth when require_auth is enabled."""
    if settings.auth.require and request.url.path not in ("/health", "/docs", "/openapi.json"):
        auth_header = request.headers.get("Authorization", "")
        expected = f"Bearer {settings.auth.token}"
        if not settings.auth.token or auth_header != expected:
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    return await call_next(request)


@app.middleware("http")
async def add_request_id(request, call_next):
    """Stamp every request with a unique ID for cross-module log tracing."""
    request_id = request.headers.get("X-Request-ID") or str(uuid4())[:12]
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


app.include_router(vision.router, prefix="/v1")
app.include_router(models_api.router, prefix="/v1")
app.include_router(playwright_api.router, prefix="/v1")
app.include_router(research_api.router, prefix="/v1")
app.include_router(desktop_api.router, prefix="/v1")
app.include_router(gate_api.router, prefix="/gate")
app.include_router(apu_api.router, prefix="/v1")
app.include_router(modules_api.router, prefix="/v1")
app.include_router(click_api.router)

# AlchemyVoice routes (chat, voice control, callbacks)
from alchemy.voice.api import callbacks as voice_callbacks
from alchemy.voice.api import chat as voice_chat
from alchemy.voice.api import voice as voice_control

app.include_router(voice_callbacks.router, prefix="/v1")
app.include_router(voice_chat.router, prefix="/v1")
app.include_router(voice_control.router, prefix="/v1")


@app.get("/health")
async def health():
    ollama_ok = getattr(app.state, "ollama_client", None) is not None
    pw_ok = getattr(app.state, "pw_agent", None) is not None
    browser_ok = getattr(app.state, "browser_manager", None) is not None
    gate_ok = getattr(app.state, "gate_reviewer", None) is not None
    desktop_agent = getattr(app.state, "desktop_agent", None)
    desktop_ok = desktop_agent is not None
    desktop_mode = desktop_agent._controller.mode if desktop_agent else None
    gui_actor_ok = getattr(app.state, "gui_actor_client", None) is not None
    orchestrator_ok = getattr(app.state, "orchestrator", None) is not None
    voice_system = getattr(app.state, "voice_system", None)
    voice_ok = voice_system is not None and voice_system.is_running
    connect = getattr(app.state, "connect", None)
    connect_ok = connect is not None
    return {
        "status": "ok", "version": __version__,
        "ollama_connected": ollama_ok,
        "playwright_agent": pw_ok, "browser_ready": browser_ok,
        "research_enabled": settings.research.enabled,
        "gate_enabled": gate_ok,
        "desktop_agent": desktop_ok,
        "desktop_mode": desktop_mode,
        "gui_actor": gui_actor_ok,
        "gpu_orchestrator": orchestrator_ok,
        "voice_enabled": voice_ok,
        "connect_enabled": connect_ok,
        "connect_devices": connect.connected_devices if connect else 0,
        "vision_model": settings.pw_escalation.model,
    }
