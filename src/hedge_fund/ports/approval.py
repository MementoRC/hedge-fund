"""Approval port: show a proposal to a human and get an explicit yes or no."""

from __future__ import annotations

from typing import Protocol


class ApprovalGate(Protocol):
    """Human approval channel. Implementations live in adapters/approval/."""

    def confirm(self, summary: str) -> bool:
        """Show ``summary`` and return True only on an explicit approval."""
