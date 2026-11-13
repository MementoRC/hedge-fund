"""Hand-built domain objects for the viz tests."""

from __future__ import annotations

import dataclasses
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

from hedge_fund.domain.account import AccountSnapshot
from hedge_fund.domain.cycle import CycleOutcome, CycleReport
from hedge_fund.domain.portfolio import Position, TargetPortfolio
from hedge_fund.domain.risk import NavHistory, NavPoint
from hedge_fund.domain.shortlist import Side

DAY0 = date(2024, 1, 2)


def make_nav(*values: float, start: date = DAY0) -> NavHistory:
    return tuple(NavPoint(date=start + timedelta(days=i), nav=v) for i, v in enumerate(values))


def make_snapshots(*values: float, start: date = DAY0) -> tuple[AccountSnapshot, ...]:
    return tuple(
        AccountSnapshot(
            as_of=point.date,
            observed_at=datetime.combine(point.date, time(21, 30), tzinfo=UTC),
            equity=point.nav,
            holdings=(),
        )
        for point in make_nav(*values, start=start)
    )


def make_position(ticker: str, sector: str, weight: float) -> Position:
    side = Side.LONG if weight > 0 else Side.SHORT
    return Position(ticker=ticker, sector=sector, side=side, weight=weight, alpha=0.0, beta=1.0)


def make_target(as_of: date, *positions: Position) -> TargetPortfolio:
    return TargetPortfolio(
        as_of=as_of,
        positions=positions,
        excluded={},
        gross=sum(abs(p.weight) for p in positions),
        net=sum(p.weight for p in positions),
        beta=0.0,
        expected_vol=0.1,
    )


def make_cycle(as_of: date = DAY0, **changes: Any) -> CycleReport:
    base = CycleReport(
        as_of=as_of,
        started_at=datetime.combine(as_of, time(21, 30), tzinfo=UTC),
        dry_run=False,
        outcome=CycleOutcome.SUBMITTED,
        failed_stage=None,
        error=None,
        account=None,
        price_unresolved={},
        fundamentals_unresolved={},
        shortlist=None,
        research=None,
        research_reused=False,
        target=None,
        verdict=None,
        order_prices=None,
        plan=None,
        decision=None,
        execution=None,
        execution_interrupted=False,
        planned_client_order_ids=(),
    )
    return dataclasses.replace(base, **changes)
