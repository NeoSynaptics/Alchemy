"""Alchemy FastAPI server — model routing + management API on port 8000."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from alchemy.agent.task_manager import TaskManager
from alchemy.api import models_api, shadow, vision
from alchemy.models.ollama_client import OllamaClient
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
    await ollama.close()
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

app.include_router(vision.router)
app.include_router(shadow.router)
app.include_router(models_api.router)


@app.get("/health")
async def health():
    wsl_ok = getattr(app.state, "shadow_controller", None) is not None
    ollama_ok = getattr(app.state, "ollama_client", None) is not None
    return {
        "status": "ok", "version": "0.2.0",
        "wsl_available": wsl_ok, "ollama_connected": ollama_ok,
    }
