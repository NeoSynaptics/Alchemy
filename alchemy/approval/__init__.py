"""Approval gate — detect irreversible actions and pause for human confirmation."""

from alchemy.approval.gate import ApprovalGate, is_irreversible

__all__ = ["ApprovalGate", "is_irreversible"]
