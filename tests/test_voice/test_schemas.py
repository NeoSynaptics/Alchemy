"""Schema roundtrip tests — verify all models serialize/deserialize correctly."""

from datetime import datetime, timezone
from uuid import uuid4

from alchemy.voice.schemas import (
    ActionTier,
    ApprovalDecision,
    ApprovalDecisionResponse,
    ApprovalRequest,
    ApprovalRequestAck,
    ModelInfo,
    ModelsResponse,
    NotifyAck,
    NotifyRequest,
    TaskStatus,
    TaskStatusResponse,
    TaskUpdateAck,
    TaskUpdateRequest,
    VisionAction,
    VisionAnalyzeRequest,
    VisionAnalyzeResponse,
    VisionTaskRequest,
    VisionTaskResponse,
)


def _roundtrip(model_instance):
    """Serialize to JSON and back, verify equality."""
    cls = type(model_instance)
    json_str = model_instance.model_dump_json()
    restored = cls.model_validate_json(json_str)
    assert restored == model_instance


class TestVisionSchemas:
    def test_vision_action(self):
        _roundtrip(VisionAction(action="click", x=100, y=200, tier=ActionTier.AUTO))

    def test_vision_task_request(self):
        _roundtrip(VisionTaskRequest(goal="send email"))

    def test_vision_task_response(self):
        _roundtrip(VisionTaskResponse(
            task_id=uuid4(), status=TaskStatus.PENDING,
            created_at=datetime.now(timezone.utc),
        ))

    def test_vision_analyze_request(self):
        _roundtrip(VisionAnalyzeRequest(screenshot_b64="abc123", goal="find button"))

    def test_vision_analyze_response(self):
        _roundtrip(VisionAnalyzeResponse(
            action=VisionAction(action="click", x=10, y=20),
            model="ui-tars:72b", inference_ms=1500.0,
        ))

    def test_task_status_response(self):
        now = datetime.now(timezone.utc)
        _roundtrip(TaskStatusResponse(
            task_id=uuid4(), status=TaskStatus.RUNNING,
            current_step=3, created_at=now, updated_at=now,
        ))

    def test_approval_decision(self):
        _roundtrip(ApprovalDecision(decided_by="user", reason="ok"))

    def test_approval_decision_response(self):
        _roundtrip(ApprovalDecisionResponse(
            task_id=uuid4(), decision="approved", status=TaskStatus.RUNNING,
        ))


class TestModelsSchemas:
    def test_model_info(self):
        _roundtrip(ModelInfo(name="ui-tars:72b", loaded=True, size_gb=42.0))

    def test_models_response(self):
        _roundtrip(ModelsResponse(
            models=[ModelInfo(name="ui-tars:72b", loaded=False, size_gb=42.0)],
            total_ram_gb=128.0, available_ram_gb=86.0,
        ))


class TestCallbackSchemas:
    def test_approval_request(self):
        _roundtrip(ApprovalRequest(
            task_id=uuid4(),
            action=VisionAction(action="click", x=100, y=200, tier=ActionTier.APPROVE),
            screenshot_b64="abc", step=3, goal="send email",
        ))

    def test_approval_request_ack(self):
        _roundtrip(ApprovalRequestAck(received=True, task_id=uuid4()))

    def test_notify_request(self):
        _roundtrip(NotifyRequest(
            task_id=uuid4(),
            action=VisionAction(action="click", x=50, y=50, tier=ActionTier.NOTIFY),
            message="Opening Firefox", step=1,
        ))

    def test_notify_ack(self):
        _roundtrip(NotifyAck(received=True))

    def test_task_update_request(self):
        _roundtrip(TaskUpdateRequest(
            task_id=uuid4(), status=TaskStatus.COMPLETED, current_step=5,
        ))

    def test_task_update_ack(self):
        _roundtrip(TaskUpdateAck(received=True))
