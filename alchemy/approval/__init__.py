"""Approval gate — detect irreversible actions and pause for human confirmation."""

from alchemy.core import ApprovalGate, is_irreversible

__all__ = ["ApprovalGate", "is_irreversible"]
