"""Alchemy FastAPI server — model routing + management API on port 8000."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from alchemy.api import models_api, shadow, vision

app = FastAPI(
    title="Alchemy",
    description="Local-first LLM core engine",
    version="0.1.0",
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
    return {"status": "ok", "version": "0.1.0"}
