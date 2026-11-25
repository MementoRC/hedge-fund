"""The backtest's approval gate approves every plan it is shown."""

from __future__ import annotations

from hedge_fund.adapters.approval.auto import AutoApprovalGate
from hedge_fund.ports.approval import ApprovalGate


def test_auto_gate_confirms_every_summary():
    gate = AutoApprovalGate()
    assert gate.confirm("buy 10 AAA") is True
    assert gate.confirm("") is True


def test_auto_gate_is_an_approval_gate():
    gate: ApprovalGate = AutoApprovalGate()
    assert gate.confirm("x") is True
