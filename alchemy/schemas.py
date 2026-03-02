"""Alchemy ↔ NEO-TX API contract — shared Pydantic models.

Both repos maintain identical copies of these schemas. Any change here
must be mirrored in neotx/schemas.py.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    DENIED = "denied"


class ActionTier(str, Enum):
    AUTO = "auto"
    NOTIFY = "notify"
    APPROVE = "approve"


class ShadowStatus(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Vision (NEO-TX → Alchemy)
# ---------------------------------------------------------------------------

class VisionAction(BaseModel):
    action: str  # click, double_click, right_click, drag, type, hotkey, scroll, wait, done, fail
    x: int | None = None
    y: int | None = None
    end_x: int | None = None       # drag end position
    end_y: int | None = None       # drag end position
    text: str | None = None        # for type / hotkey / finished content
    reasoning: str = ""
    tier: ActionTier = ActionTier.AUTO
    direction: str | None = None   # scroll direction: up/down/left/right
    amount: int | None = None      # scroll amount


class VisionTaskRequest(BaseModel):
    goal: str
    context: dict | None = None
    callback_url: str = "http://localhost:8100"


class VisionTaskResponse(BaseModel):
    task_id: UUID
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime


class VisionAnalyzeRequest(BaseModel):
    screenshot_b64: str
    goal: str
    context: dict | None = None


class VisionAnalyzeResponse(BaseModel):
    action: VisionAction
    model: str = "ui-tars:72b"
    inference_ms: float = 0.0


class TaskStatusResponse(BaseModel):
    task_id: UUID
    status: TaskStatus
    current_step: int = 0
    total_steps: int | None = None
    last_action: VisionAction | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class ApprovalDecision(BaseModel):
    decided_by: str = "user"
    reason: str | None = None


class ApprovalDecisionResponse(BaseModel):
    task_id: UUID
    decision: str  # "approved" or "denied"
    status: TaskStatus


# ---------------------------------------------------------------------------
# Shadow Desktop (NEO-TX → Alchemy)
# ---------------------------------------------------------------------------

class ShadowStartRequest(BaseModel):
    resolution: str = "1920x1080x24"
    display_num: int = 99


class ShadowStartResponse(BaseModel):
    status: ShadowStatus
    display: str  # ":99"
    vnc_url: str  # "localhost:5900"
    novnc_url: str  # "http://localhost:6080/vnc.html?autoconnect=true"


class ShadowStopResponse(BaseModel):
    status: ShadowStatus
    message: str = "Shadow desktop stopped"


class ShadowHealthResponse(BaseModel):
    status: ShadowStatus
    xvfb_running: bool = False
    fluxbox_running: bool = False
    vnc_running: bool = False
    novnc_running: bool = False
    display: str = ":99"
    uptime_seconds: float | None = None


# ---------------------------------------------------------------------------
# Models (NEO-TX → Alchemy)
# ---------------------------------------------------------------------------

class ModelInfo(BaseModel):
    name: str
    loaded: bool
    size_gb: float
    ram_used_gb: float | None = None


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    total_ram_gb: float
    available_ram_gb: float


# ---------------------------------------------------------------------------
# Callbacks (Alchemy → NEO-TX)
# ---------------------------------------------------------------------------

class ApprovalRequest(BaseModel):
    task_id: UUID
    action: VisionAction
    screenshot_b64: str
    step: int
    timeout_seconds: int = 60
    goal: str


class ApprovalRequestAck(BaseModel):
    received: bool = True
    task_id: UUID


class NotifyRequest(BaseModel):
    task_id: UUID
    action: VisionAction
    message: str
    step: int
    screenshot_b64: str | None = None


class NotifyAck(BaseModel):
    received: bool = True


class TaskUpdateRequest(BaseModel):
    task_id: UUID
    status: TaskStatus
    current_step: int
    total_steps: int | None = None
    last_action: VisionAction | None = None
    message: str | None = None
    error: str | None = None


class TaskUpdateAck(BaseModel):
    received: bool = True
