"""Approval types: the human decision on a risk-assessed order plan."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum


class ApprovalStatus(StrEnum):
    """Outcome of one approval request.

    APPROVED: the human typed y / yes. REJECTED: the human declined or gave no answer (no
    terminal, EOF, Ctrl-C). BLOCKED: the plan is blocked (halt / veto) and was never
    offered. NOTHING_TO_DO: the plan has no orders and was never offered.
    """

    APPROVED = "approved"
    REJECTED = "rejected"
    BLOCKED = "blocked"
    NOTHING_TO_DO = "nothing-to-do"


@dataclass(frozen=True)
class ApprovalDecision:
    """The decision on one plan.

    ``plan_fingerprint`` identifies the orders that were decided on (see
    ``engine.approval.plan_fingerprint``); ``summary`` is the text shown to the human, or
    that would have been shown for a plan never offered.
    """

    as_of: date
    status: ApprovalStatus
    plan_fingerprint: str
    summary: str
