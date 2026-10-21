"""Order generation: validation, veto, sizing, legs, ordering."""

from __future__ import annotations

import dataclasses
import math
from datetime import date

import pytest

from hedge_fund.domain.orders import (
    BLOCKED_HALT,
    BLOCKED_VETO,
    Holding,
    Order,
    OrderAction,
    OrderPlan,
)
from hedge_fund.domain.portfolio import Position, TargetPortfolio
from hedge_fund.domain.risk import BreakerState, RiskBreakdown, RiskVerdict
from hedge_fund.domain.shortlist import Side
from hedge_fund.engine.orders.generate import generate_orders
from hedge_fund.engine.orders.params import OrderParams

AS_OF = date(2026, 10, 21)
NAV = 1_000_000.0
PRICE = 100.0  # one share is 1e-4 of NAV
QUIET = BreakerState(daily_loss=0.0, drawdown=0.0, daily_tripped=False, drawdown_tripped=False)
BREAKDOWN = RiskBreakdown(
    total_vol=0.05,
    factor_vol=0.0,
    specific_vol=0.0,
    factor_share=0.0,
    by_factor={},
    by_name={},
    unassessed=(),
)
BUY = OrderAction.BUY
SELL = OrderAction.SELL
SELL_SHORT = OrderAction.SELL_SHORT
BUY_TO_COVER = OrderAction.BUY_TO_COVER


def _verdict(*, approved: bool = True, halt: bool = False, as_of: date = AS_OF) -> RiskVerdict:
    return RiskVerdict(
        as_of=as_of,
        approved=approved,
        halt=halt,
        violations=(),
        breakdown=BREAKDOWN,
        breakers=QUIET,
    )


def _target(
    weights: dict[str, float], *, excluded: dict[str, str] | None = None
) -> TargetPortfolio:
    positions = tuple(
        Position(
            ticker=ticker,
            sector="Tech",
            side=Side.SHORT if weight < 0 else Side.LONG,
            weight=weight,
            alpha=0.0,
            beta=1.0,
        )
        for ticker, weight in weights.items()
    )
    return TargetPortfolio(
        as_of=AS_OF,
        positions=positions,
        excluded=excluded or {},
        gross=sum(abs(weight) for weight in weights.values()),
        net=sum(weights.values()),
        beta=0.0,
        expected_vol=0.0,
    )


def _run(
    weights: dict[str, float],
    held: dict[str, int],
    *,
    prices: dict[str, float] | None = None,
    nav: float = NAV,
    band: float = 0.0025,
    verdict: RiskVerdict | None = None,
    excluded: dict[str, str] | None = None,
) -> OrderPlan:
    tickers = weights.keys() | held.keys()
    return generate_orders(
        _target(weights, excluded=excluded),
        _verdict() if verdict is None else verdict,
        tuple(Holding(ticker=ticker, shares=shares) for ticker, shares in held.items()),
        dict.fromkeys(tickers, PRICE) if prices is None else prices,
        nav,
        params=OrderParams(no_trade_band=band),
    )


def _order(ticker: str, action: OrderAction, shares: int, price: float = PRICE) -> Order:
    return Order(
        ticker=ticker, action=action, shares=shares, ref_price=price, notional=shares * price
    )


# --- validation -------------------------------------------------------------------------


def test_rejects_invalid_params() -> None:
    with pytest.raises(ValueError, match="no_trade_band must be a finite number"):
        _run({"A": 0.125}, {}, band=math.nan)


def test_rejects_as_of_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match target as_of"):
        _run({"A": 0.125}, {}, verdict=_verdict(as_of=date(2026, 10, 20)))


@pytest.mark.parametrize("nav", [math.nan, math.inf, 0.0, -1.0, True, "1e6"])
def test_rejects_invalid_nav(nav: object) -> None:
    with pytest.raises(ValueError, match="nav must be a finite number > 0"):
        _run({"A": 0.125}, {}, nav=nav)  # type: ignore[arg-type]


@pytest.mark.parametrize("weight", [math.nan, math.inf, -math.inf, True])
def test_rejects_invalid_weight(weight: float) -> None:
    with pytest.raises(ValueError, match="weight must be a finite number, got .* for A"):
        _run({"A": weight}, {})


def test_rejects_duplicate_position() -> None:
    target = _target({"A": 0.125})
    doubled = dataclasses.replace(target, positions=target.positions * 2)
    with pytest.raises(ValueError, match="duplicate position ticker 'A'"):
        generate_orders(doubled, _verdict(), (), {"A": PRICE}, NAV)


@pytest.mark.parametrize("shares", [1.0, True, "1"])
def test_rejects_non_int_shares(shares: int) -> None:
    with pytest.raises(ValueError, match="shares must be an int, got .* for A"):
        _run({}, {"A": shares})


def test_rejects_duplicate_holding() -> None:
    holdings = (Holding(ticker="A", shares=1), Holding(ticker="A", shares=2))
    with pytest.raises(ValueError, match="duplicate holding ticker 'A'"):
        generate_orders(_target({}), _verdict(), holdings, {"A": PRICE}, NAV)


@pytest.mark.parametrize("price", [math.nan, math.inf, 0.0, -1.0, True, None])
def test_rejects_invalid_price(price: float) -> None:
    with pytest.raises(ValueError, match="price for A must be a finite number > 0"):
        _run({"A": 0.125}, {}, prices={"A": price})


def test_rejects_missing_price_for_target_name() -> None:
    with pytest.raises(ValueError, match="price for A must be a finite number > 0, got None"):
        _run({"A": 0.125}, {}, prices={})


def test_rejects_missing_price_for_held_only_name() -> None:
    with pytest.raises(ValueError, match="price for B must be a finite number > 0, got None"):
        _run({}, {"B": 10}, prices={})


def test_validation_runs_before_veto() -> None:
    with pytest.raises(ValueError, match="nav must be a finite number > 0"):
        _run({"A": 0.125}, {}, nav=math.nan, verdict=_verdict(approved=False, halt=True))


# --- veto -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("approved", "halt", "reason"),
    [
        (True, True, BLOCKED_HALT),
        (False, False, BLOCKED_VETO),
        (False, True, BLOCKED_HALT),
    ],
)
def test_blocked_plan(approved: bool, halt: bool, reason: str) -> None:
    plan = _run(
        {"A": 0.0625, "B": 0.125},
        {"A": 1250, "C": 0},
        verdict=_verdict(approved=approved, halt=halt),
    )
    assert plan == OrderPlan(
        as_of=AS_OF,
        orders=(),
        skipped={},
        residual_weight={"A": 0.0625, "B": -0.125, "C": 0.0},
        blocked=reason,
    )
    assert list(plan.residual_weight) == ["A", "B", "C"]
