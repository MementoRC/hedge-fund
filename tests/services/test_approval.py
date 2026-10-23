"""Layer 6c approval service: decision rules over a scripted gate."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from hedge_fund.domain.approval import ApprovalDecision, ApprovalStatus
from hedge_fund.domain.orders import BLOCKED_HALT, BLOCKED_VETO, Order, OrderAction, OrderPlan
from hedge_fund.domain.risk import BreakerState, RiskBreakdown, RiskVerdict, Violation
from hedge_fund.engine.approval import plan_fingerprint, render_proposal
from hedge_fund.services.approval import request_approval
from tests.fakes import ScriptedApprovalGate

AS_OF = date(2026, 10, 23)
SELL_A = Order(ticker="A", action=OrderAction.SELL, shares=10, ref_price=100.0, notional=1000.0)
BUY_B = Order(ticker="B", action=OrderAction.BUY, shares=5, ref_price=20.0, notional=100.0)
GROSS = Violation(check="gross", observed=1.6, limit=1.5, detail="")


def _verdict(
    *, violations: tuple[Violation, ...] = (), halt: bool = False, as_of: date = AS_OF
) -> RiskVerdict:
    return RiskVerdict(
        as_of=as_of,
        approved=not violations and not halt,
        halt=halt,
        violations=violations,
        breakdown=RiskBreakdown(
            total_vol=0.1,
            factor_vol=0.08,
            specific_vol=0.06,
            factor_share=0.64,
            by_factor={"value": 0.0064},
            by_name={"A": 0.7, "B": 0.3},
            unassessed=(),
        ),
        breakers=BreakerState(
            daily_loss=-0.01, drawdown=0.02, daily_tripped=False, drawdown_tripped=halt
        ),
    )


def _plan(*orders: Order, blocked: str | None = None) -> OrderPlan:
    return OrderPlan(
        as_of=AS_OF, orders=orders, skipped={}, residual_weight={"A": 0.001}, blocked=blocked
    )


APPROVED_VERDICT = _verdict()
VETO_VERDICT = _verdict(violations=(GROSS,))
HALT_VERDICT = _verdict(halt=True)
HALT_AND_VETO_VERDICT = _verdict(violations=(GROSS,), halt=True)


def _decision(verdict: RiskVerdict, plan: OrderPlan, status: ApprovalStatus) -> ApprovalDecision:
    return ApprovalDecision(
        as_of=AS_OF,
        status=status,
        plan_fingerprint=plan_fingerprint(plan),
        summary=render_proposal(verdict, plan),
    )


# --- decision table ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "status"),
    [(True, ApprovalStatus.APPROVED), (False, ApprovalStatus.REJECTED)],
)
def test_plan_with_orders_is_offered(answer: bool, status: ApprovalStatus) -> None:
    plan = _plan(SELL_A, BUY_B)
    gate = ScriptedApprovalGate([answer])
    decision = request_approval(APPROVED_VERDICT, plan, gate)
    assert decision == _decision(APPROVED_VERDICT, plan, status)
    assert gate.summaries == [decision.summary]


@pytest.mark.parametrize(
    ("verdict", "blocked"),
    [
        pytest.param(VETO_VERDICT, BLOCKED_VETO, id="veto"),
        pytest.param(HALT_VERDICT, BLOCKED_HALT, id="halt"),
        pytest.param(HALT_AND_VETO_VERDICT, BLOCKED_HALT, id="halt-outranks-veto"),
    ],
)
def test_blocked_plan_is_never_offered(verdict: RiskVerdict, blocked: str) -> None:
    plan = _plan(blocked=blocked)
    gate = ScriptedApprovalGate([])
    decision = request_approval(verdict, plan, gate)
    assert decision == _decision(verdict, plan, ApprovalStatus.BLOCKED)
    assert gate.summaries == []


def test_empty_plan_is_never_offered() -> None:
    plan = _plan()
    gate = ScriptedApprovalGate([])
    decision = request_approval(APPROVED_VERDICT, plan, gate)
    assert decision == _decision(APPROVED_VERDICT, plan, ApprovalStatus.NOTHING_TO_DO)
    assert gate.summaries == []


# --- mismatches and malformed plans -----------------------------------------------------


def test_as_of_mismatch_raises() -> None:
    gate = ScriptedApprovalGate([])
    with pytest.raises(ValueError, match="verdict as_of 2026-10-24 does not match plan as_of"):
        request_approval(_verdict(as_of=date(2026, 10, 24)), _plan(SELL_A), gate)
    assert gate.summaries == []


@pytest.mark.parametrize(
    ("verdict", "blocked", "expected"),
    [
        pytest.param(APPROVED_VERDICT, BLOCKED_VETO, "None", id="approved-but-blocked"),
        pytest.param(VETO_VERDICT, None, "'veto'", id="vetoed-but-unblocked"),
        pytest.param(HALT_VERDICT, BLOCKED_VETO, "'halt'", id="halt-marked-veto"),
        pytest.param(HALT_AND_VETO_VERDICT, BLOCKED_VETO, "'halt'", id="halt-outranks-veto"),
        pytest.param(VETO_VERDICT, BLOCKED_HALT, "'veto'", id="veto-marked-halt"),
    ],
)
def test_blocked_mismatch_raises(verdict: RiskVerdict, blocked: str | None, expected: str) -> None:
    gate = ScriptedApprovalGate([])
    with pytest.raises(ValueError, match=f"does not match the verdict \\({expected}\\)"):
        request_approval(verdict, _plan(blocked=blocked), gate)
    assert gate.summaries == []


def test_malformed_plan_raises_before_the_gate() -> None:
    gate = ScriptedApprovalGate([True])
    with pytest.raises(ValueError, match="duplicate client_order_id"):
        request_approval(APPROVED_VERDICT, _plan(BUY_B, BUY_B), gate)
    assert gate.summaries == []


# --- gate behaviour ---------------------------------------------------------------------


class _AnswerGate:
    """Gate returning a fixed value, which a buggy adapter might make a non-bool."""

    def __init__(self, answer: Any) -> None:
        self.answer = answer

    def confirm(self, summary: str) -> bool:
        return self.answer


@pytest.mark.parametrize("answer", [1, "yes", None], ids=["one", "yes", "none"])
def test_only_true_approves(answer: Any) -> None:
    decision = request_approval(APPROVED_VERDICT, _plan(SELL_A), _AnswerGate(answer))
    assert decision.status == ApprovalStatus.REJECTED


class _BrokenGate:
    def confirm(self, summary: str) -> bool:
        raise OSError("terminal gone")


def test_gate_exceptions_propagate() -> None:
    with pytest.raises(OSError, match="terminal gone"):
        request_approval(APPROVED_VERDICT, _plan(SELL_A), _BrokenGate())
