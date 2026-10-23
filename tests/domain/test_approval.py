"""Approval domain types."""

from __future__ import annotations

import dataclasses
from datetime import date

import pytest

from hedge_fund.domain.approval import ApprovalDecision, ApprovalStatus


def test_approval_status_values() -> None:
    assert [status.value for status in ApprovalStatus] == [
        "approved",
        "rejected",
        "blocked",
        "nothing-to-do",
    ]
    assert ApprovalStatus.NOTHING_TO_DO == "nothing-to-do"


def test_decision_is_frozen() -> None:
    decision = ApprovalDecision(
        as_of=date(2026, 10, 23),
        status=ApprovalStatus.REJECTED,
        plan_fingerprint="0" * 64,
        summary="Proposal\n",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.status = ApprovalStatus.APPROVED
