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

from config.logging import setup_logging
from fastapi import FastAPI

setup_logging()
from fastapi.middleware.cors import CORSMiddleware

from alchemy.adapters import OllamaClient
from alchemy.click.task_manager import TaskManager
from alchemy.api import models_api, shadow, vision
from alchemy.api import playwright_api
from alchemy.api import research_api
from alchemy.api import gate_api
from alchemy.api import desktop_api
from alchemy.api import gpu_api
from alchemy.api import modules_api
from alchemy.router.environment import EnvironmentDetector
from alchemy.shadow.controller import ShadowDesktopController
from alchemy.shadow.wsl import WslRunner
from config.settings import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shadow desktop controller on startup."""
    wsl = WslRunner(distro=settings.wsl_distro, display_num=settings.display_num)

    # Run sync WSL check in thread to avoid blocking the event loop
    wsl_ok = await asyncio.to_thread(wsl.is_available)

    if wsl_ok:
        logger.info("WSL2 (%s) available — shadow desktop controller active", settings.wsl_distro)
        app.state.shadow_controller = ShadowDesktopController(
            wsl=wsl,
            display_num=settings.display_num,
            vnc_port=settings.vnc_port,
            novnc_port=settings.novnc_port,
            resolution=settings.resolution,
            screenshot_format=settings.screenshot_format,
            screenshot_jpeg_quality=settings.screenshot_jpeg_quality,
            screenshot_resize_width=settings.screenshot_resize_width,
            screenshot_resize_height=settings.screenshot_resize_height,
            repo_wsl_path=settings.shadow_wsl_repo_path,
        )
    else:
        logger.warning("WSL2 not available — shadow desktop endpoints will return mock data")
        app.state.shadow_controller = None

    # Initialize Ollama client
    ollama = OllamaClient(
        host=settings.ollama_host,
        timeout=120.0,
        keep_alive=settings.ollama_keep_alive,
        retry_attempts=settings.ollama_retry_attempts,
        retry_delay=settings.ollama_retry_delay,
    )
    await ollama.start()

    if await ollama.ping():
        logger.info("Ollama at %s — connected", settings.ollama_host)
        # Check both models
        for model_name in [settings.ollama_cpu_model, settings.ollama_fast_model]:
            if await ollama.is_model_available(model_name):
                logger.info("Model %s — available", model_name)
            else:
                logger.warning("Model %s not found — pull with: ollama pull %s",
                              model_name, model_name)
    else:
        logger.warning("Ollama not reachable at %s", settings.ollama_host)

    app.state.ollama_client = ollama
    app.state.task_manager = TaskManager()

    # --- GPU Stack Orchestrator ---
    try:
        from pathlib import Path
        from alchemy.gpu import GPUMonitor, ModelRegistry, StackOrchestrator

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
        logger.info("GPU Stack Orchestrator started (%d models registered)",
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
        logger.warning("GPU Stack Orchestrator failed to start: %s", e)
        app.state.orchestrator = None
        app.state.gpu_monitor = None
        app.state.model_registry = None
        app.state.contract_reports = {}

    # --- Playwright Agent (Tier 1) ---
    if settings.pw_enabled:
        try:
            from alchemy.core import PlaywrightAgent, ApprovalGate, BrowserManager

            browser_mgr = BrowserManager(headless=settings.pw_headless)
            await browser_mgr.start()
            app.state.browser_manager = browser_mgr

            approval_gate = ApprovalGate(enabled=settings.pw_approval_enabled)

            # Tier 1.5: Vision escalation (UI-TARS 7B fallback when stuck)
            vision_escalation = None
            stuck_detector = None
            if settings.pw_escalation_enabled:
                from alchemy.core import StuckDetector, VisionEscalation

                vision_escalation = VisionEscalation(
                    ollama_client=ollama,
                    model=settings.pw_escalation_model,
                    temperature=settings.pw_escalation_temperature,
                    max_tokens=settings.pw_escalation_max_tokens,
                )
                stuck_detector = StuckDetector(
                    max_parse_failures=settings.pw_escalation_parse_failures,
                    max_repeated_actions=settings.pw_escalation_repeated_actions,
                    complexity_threshold=settings.pw_escalation_complexity_threshold,
                )
                logger.info("Tier 1.5 escalation ready (model=%s)", settings.pw_escalation_model)

            pw_agent = PlaywrightAgent(
                ollama_client=ollama,
                model=settings.pw_model,
                max_steps=settings.pw_max_steps,
                think=settings.pw_think,
                temperature=settings.pw_temperature,
                max_tokens=settings.pw_max_tokens,
                settle_timeout=settings.pw_settle_timeout,
                approval_checker=lambda action: approval_gate.needs_approval(action),
                vision_escalation=vision_escalation,
                stuck_detector=stuck_detector,
            )
            app.state.pw_agent = pw_agent
            logger.info("Playwright agent ready (model=%s, think=%s, escalation=%s)",
                        settings.pw_model, settings.pw_think, settings.pw_escalation_enabled)

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
    if settings.desktop_enabled:
        try:
            from alchemy.desktop import DesktopAgent, DesktopController

            desktop_ctrl = DesktopController(
                screenshot_width=settings.desktop_screenshot_width,
                screenshot_height=settings.desktop_screenshot_height,
                screenshot_quality=settings.desktop_screenshot_quality,
                mode=settings.desktop_default_mode,
            )
            app.state.desktop_agent = DesktopAgent(
                ollama_client=ollama,
                controller=desktop_ctrl,
                model=settings.desktop_model,
                max_steps=settings.desktop_max_steps,
                temperature=settings.desktop_temperature,
                max_tokens=settings.desktop_max_tokens,
            )
            logger.info("Desktop agent ready (model=%s)", settings.desktop_model)
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
    if settings.gui_actor_enabled:
        try:
            from alchemy.adapters import GUIActorClient

            gui_actor = GUIActorClient(
                host=settings.gui_actor_host,
                timeout=settings.gui_actor_timeout,
            )
            await gui_actor.start()
            if await gui_actor.ping():
                app.state.gui_actor_client = gui_actor
                logger.info("GUI-Actor client ready (host=%s)", settings.gui_actor_host)
            else:
                logger.warning("GUI-Actor server not reachable at %s", settings.gui_actor_host)
                await gui_actor.close()
        except Exception as e:
            logger.warning("GUI-Actor client failed to start: %s", e)

    # --- Gate (Claude Code auto-approve) ---
    if settings.gate_enabled:
        from alchemy.gate import GateReviewer

        app.state.gate_reviewer = GateReviewer(
            ollama_client=ollama,
            model=settings.gate_model,
            timeout=settings.gate_timeout,
        )
        logger.info("Gate reviewer ready (model=%s)", settings.gate_model)
    else:
        app.state.gate_reviewer = None

    # Detect environment for context router
    if settings.router_enabled:
        detector = EnvironmentDetector(wsl=wsl if wsl_ok else None)
        app.state.environment = await detector.detect()
        logger.info(
            "Router environment: %d shadow apps, %d windows apps",
            len(app.state.environment.shadow_apps),
            len(app.state.environment.windows_apps),
        )
    else:
        app.state.environment = None

    yield

    # Cleanup
    if getattr(app.state, "orchestrator", None):
        await app.state.orchestrator.close()
        logger.info("GPU Stack Orchestrator stopped")

    if getattr(app.state, "browser_manager", None):
        await app.state.browser_manager.close()
        logger.info("Playwright browser closed")

    await ollama.close()
    if getattr(app.state, "gui_actor_client", None):
        await app.state.gui_actor_client.close()
    if app.state.shadow_controller:
        logger.info("Shutting down shadow desktop...")
        await app.state.shadow_controller.stop()


app = FastAPI(
    title="Alchemy",
    description="Local-first LLM core engine",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request, call_next):
    """Stamp every request with a unique ID for cross-module log tracing."""
    request_id = request.headers.get("X-Request-ID") or str(uuid4())[:12]
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


app.include_router(vision.router, prefix="/v1")
app.include_router(shadow.router, prefix="/v1")
app.include_router(models_api.router, prefix="/v1")
app.include_router(playwright_api.router, prefix="/v1")
app.include_router(research_api.router, prefix="/v1")
app.include_router(desktop_api.router, prefix="/v1")
app.include_router(gate_api.router, prefix="/gate")
app.include_router(gpu_api.router, prefix="/v1")
app.include_router(modules_api.router, prefix="/v1")


@app.get("/health")
async def health():
    wsl_ok = getattr(app.state, "shadow_controller", None) is not None
    ollama_ok = getattr(app.state, "ollama_client", None) is not None
    pw_ok = getattr(app.state, "pw_agent", None) is not None
    browser_ok = getattr(app.state, "browser_manager", None) is not None
    gate_ok = getattr(app.state, "gate_reviewer", None) is not None
    desktop_agent = getattr(app.state, "desktop_agent", None)
    desktop_ok = desktop_agent is not None
    desktop_mode = desktop_agent._controller.mode if desktop_agent else None
    gui_actor_ok = getattr(app.state, "gui_actor_client", None) is not None
    orchestrator_ok = getattr(app.state, "orchestrator", None) is not None
    return {
        "status": "ok", "version": "0.3.0",
        "wsl_available": wsl_ok, "ollama_connected": ollama_ok,
        "playwright_agent": pw_ok, "browser_ready": browser_ok,
        "research_enabled": settings.research_enabled,
        "gate_enabled": gate_ok,
        "desktop_agent": desktop_ok,
        "desktop_mode": desktop_mode,
        "gui_actor": gui_actor_ok,
        "gpu_orchestrator": orchestrator_ok,
        "vision_model": settings.pw_escalation_model,
    }
