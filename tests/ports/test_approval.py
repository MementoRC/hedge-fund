"""Tests for the ApprovalGate port, exercised through the scripted fake."""

from __future__ import annotations

import pytest

from hedge_fund.ports.approval import ApprovalGate
from tests.fakes import ScriptedApprovalGate


def test_scripted_gate_satisfies_the_port() -> None:
    fake = ScriptedApprovalGate([True, False])
    gate: ApprovalGate = fake
    assert gate.confirm("first\n") is True
    assert gate.confirm("second\n") is False
    assert fake.summaries == ["first\n", "second\n"]


def test_scripted_gate_raises_when_out_of_answers() -> None:
    fake = ScriptedApprovalGate([])
    with pytest.raises(AssertionError, match="out of answers"):
        fake.confirm("summary\n")
    assert fake.summaries == ["summary\n"]
