"""BrainPhysics API — experiment endpoints for the cognitive routing simulator."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from .engine import BrainPhysicsEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/brain-physics", tags=["brain_physics"])


@router.get("/status")
async def status(request: Request):
    """Current state of the BrainPhysics engine."""
    engine: BrainPhysicsEngine | None = getattr(request.app.state, "brain_physics_engine", None)
    return {
        "enabled": engine is not None,
        "memory_patterns": len(engine._memory) if engine else 0,
        "phase": "stub",
        "components": {
            "perceiver": "stub",       # coarse-to-fine VLM perception
            "scene_graph": "stub",     # spatial object + relation extraction
            "physics_sim": "stub",     # intuitive physics simulation
            "prediction_loop": "stub", # predictive processing error minimization
            "consolidation": "stub",   # memory distillation
        },
        "experiment_plan": {
            "phase_1": "Wire VLM perception → extract scene graph from screenshot",
            "phase_2": "Spatial relation extraction (above/below/near/inside)",
            "phase_3": "Physics simulation via VLM reasoning (what happens if I click X?)",
            "phase_4": "Prediction error loop (compare predicted vs actual scene)",
            "phase_5": "Memory consolidation (distill successful patterns)",
            "phase_6": "Benchmark: BrainPhysics router vs. flat model routing",
        },
    }


@router.post("/step")
async def step(request: Request):
    """Run one perceive→simulate→predict→refine cycle.

    Body (optional):
        { "goal": "open Chrome", "screenshot_base64": "..." }
    """
    engine: BrainPhysicsEngine | None = getattr(request.app.state, "brain_physics_engine", None)
    if not engine:
        return {"error": "BrainPhysics engine not initialized", "hint": "Set brain_physics_enabled=true"}

    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    goal = body.get("goal", "")

    result = await engine.step(screenshot=None, goal=goal)

    return {
        "iteration": result.iteration,
        "resolution": result.scene.resolution,
        "nodes": len(result.scene.nodes),
        "action": result.action.action_type if result.action else None,
        "prediction_confidence": result.prediction.confidence if result.prediction else None,
        "error_magnitude": result.error.error_magnitude if result.error else None,
        "refined": result.refined,
        "distilled": result.distilled,
        "elapsed_ms": round(result.elapsed_ms, 2),
    }
