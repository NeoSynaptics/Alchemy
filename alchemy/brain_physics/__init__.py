"""BrainPhysics — coarse-to-fine cognitive routing with intuitive physics simulation.

Unifies predictive processing, embodied cognition, coarse-to-fine perception,
and intuitive physics into a single routing architecture. The brain doesn't
brute-force — it sketches, simulates, checks, and refines.

Public API:
    BrainPhysicsEngine  — the main loop (perceive → simulate → predict → refine)
    SceneGraph          — spatial representation of objects + relations
    PhysicsSim          — lightweight intuitive physics (approximate, not analytical)
    PredictionLoop      — iterative error-minimization (predictive processing)
"""

__all__ = [
    "BrainPhysicsEngine",
    "SceneGraph",
    "PhysicsSim",
    "PredictionLoop",
]

# Lazy imports — module is a stub; classes will be implemented incrementally.
# For now, re-export from engine once it exists.
try:
    from .engine import BrainPhysicsEngine, SceneGraph, PhysicsSim, PredictionLoop
except ImportError:
    pass
