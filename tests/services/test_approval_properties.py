"""Seeded invariants of render_proposal and request_approval over generate_orders plans."""

from __future__ import annotations

import dataclasses

import pytest

from hedge_fund.domain.approval import ApprovalStatus
from hedge_fund.domain.orders import OrderPlan
from hedge_fund.domain.risk import RiskVerdict, Violation
from hedge_fund.engine.approval import render_proposal
from hedge_fund.engine.orders import OrderParams, generate_orders
from hedge_fund.services.approval import request_approval
from tests.engine.orders.test_generate_properties import NAV, VERDICT, _book

SEEDS = range(300)
VETOED = dataclasses.replace(
    VERDICT,
    approved=False,
    violations=(Violation(check="gross", observed=2.0, limit=1.5, detail=""),),
)
HALTED = dataclasses.replace(
    VERDICT,
    approved=False,
    halt=True,
    breakers=dataclasses.replace(VERDICT.breakers, daily_tripped=True),
)
VERDICTS = (VERDICT, VETOED, HALTED)


class NeverGate:
    """Gate that always declines."""

    def confirm(self, summary: str) -> bool:
        return False


def _plan(seed: int, verdict: RiskVerdict) -> OrderPlan:
    target, holdings, prices, band = _book(seed)
    return generate_orders(
        target, verdict, holdings, prices, NAV, params=OrderParams(no_trade_band=band)
    )


def _expected(plan: OrderPlan) -> ApprovalStatus:
    if plan.blocked is not None:
        return ApprovalStatus.BLOCKED
    return ApprovalStatus.REJECTED if plan.orders else ApprovalStatus.NOTHING_TO_DO


@pytest.mark.parametrize("seed", SEEDS)
def test_render_invariants(seed: int) -> None:
    for verdict in VERDICTS:
        plan = _plan(seed, verdict)
        text = render_proposal(verdict, plan)
        assert text == render_proposal(verdict, plan)
        assert text.isascii()
        assert text.endswith("\n")
        assert all(line == line.rstrip() for line in text.split("\n"))
        assert text.startswith(f"Proposal for {plan.as_of.isoformat()}   risk: ")
        table = text.split("\n\n")[1].split("\n")[2 : 2 + len(plan.orders)]
        cells = [tuple(row.split()[1:4]) for row in table]
        assert cells == [
            (order.ticker, order.action.value, str(order.shares)) for order in plan.orders
        ]


@pytest.mark.parametrize("seed", SEEDS)
def test_a_declining_gate_never_approves(seed: int) -> None:
    for verdict in VERDICTS:
        plan = _plan(seed, verdict)
        decision = request_approval(verdict, plan, NeverGate())
        assert decision.status == _expected(plan)


def test_grid_exercises_every_outcome() -> None:
    plans = [_plan(seed, verdict) for seed in SEEDS for verdict in VERDICTS]
    assert {_expected(plan) for plan in plans} == {
        ApprovalStatus.REJECTED,
        ApprovalStatus.BLOCKED,
        ApprovalStatus.NOTHING_TO_DO,
    }
