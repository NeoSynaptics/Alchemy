"""Alchemy FastAPI server — model routing + management API on port 8000."""

import asyncio
import logging
from contextlib import asynccontextmanager
from functools import partial

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from alchemy.api import models_api, shadow, vision
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
        )
    else:
        logger.warning("WSL2 not available — shadow desktop endpoints will return mock data")
        app.state.shadow_controller = None

    yield

    # Cleanup: stop shadow desktop if running
    if app.state.shadow_controller:
        logger.info("Shutting down shadow desktop...")
        await app.state.shadow_controller.stop()


app = FastAPI(
    title="Alchemy",
    description="Local-first LLM core engine",
    version="0.1.0",
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
    return {"status": "ok", "version": "0.1.0", "wsl_available": wsl_ok}
