"""Shim — escalation moved to alchemy.core.escalation."""
from alchemy.core.escalation import (  # noqa: F401
    EscalationResult,
    StuckDetector,
    StuckReason,
    VisionEscalation,
    _parse_escalation_response,
    extract_task_text,
)
