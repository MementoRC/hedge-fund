"""Seeded-grid invariants of generate_orders (no hypothesis dependency)."""

from __future__ import annotations

import math
import random
from datetime import date

import pytest

from hedge_fund.domain.orders import SKIPPED_BAND, Holding, OrderAction
from hedge_fund.domain.portfolio import Position, TargetPortfolio
from hedge_fund.domain.risk import BreakerState, RiskBreakdown, RiskVerdict
from hedge_fund.domain.shortlist import Side
from hedge_fund.engine.orders.generate import generate_orders
from hedge_fund.engine.orders.params import OrderParams

AS_OF = date(2026, 10, 21)
NAV = 1_000_000.0
TOL = 1e-12
SIGN = {
    OrderAction.BUY: 1,
    OrderAction.BUY_TO_COVER: 1,
    OrderAction.SELL: -1,
    OrderAction.SELL_SHORT: -1,
}
REDUCING = {OrderAction.SELL, OrderAction.BUY_TO_COVER}
VERDICT = RiskVerdict(
    as_of=AS_OF,
    approved=True,
    halt=False,
    violations=(),
    breakdown=RiskBreakdown(
        total_vol=0.05,
        factor_vol=0.0,
        specific_vol=0.0,
        factor_share=0.0,
        by_factor={},
        by_name={},
        unassessed=(),
    ),
    breakers=BreakerState(
        daily_loss=0.0, drawdown=0.0, daily_tripped=False, drawdown_tripped=False
    ),
)


def _book(
    seed: int,
) -> tuple[TargetPortfolio, tuple[Holding, ...], dict[str, float], float]:
    """Random book; continuous random weights and prices keep shares off exact integers."""
    rng = random.Random(seed)
    tickers = [f"T{i:02d}" for i in range(rng.randint(1, 12))]
    prices = {ticker: rng.uniform(5.0, 500.0) for ticker in tickers}
    positions: list[Position] = []
    holdings: list[Holding] = []
    for ticker in tickers:
        weight = None
        if rng.random() < 0.7:
            weight = 0.0 if rng.random() < 0.1 else rng.uniform(-0.1, 0.1)
            side = Side.SHORT if weight < 0 else Side.LONG
            positions.append(
                Position(
                    ticker=ticker, sector="Tech", side=side, weight=weight, alpha=0.0, beta=1.0
                )
            )
        if rng.random() < 0.7:
            if weight is not None and rng.random() < 0.5:  # near target: exercises the band
                shares = round(weight * NAV / prices[ticker]) + rng.randint(-30, 30)
            else:
                shares = rng.randint(-3000, 3000)
            holdings.append(Holding(ticker=ticker, shares=shares))
    rng.shuffle(positions)
    rng.shuffle(holdings)
    target = TargetPortfolio(
        as_of=AS_OF,
        positions=tuple(positions),
        excluded={},
        gross=sum(abs(position.weight) for position in positions),
        net=sum(position.weight for position in positions),
        beta=0.0,
        expected_vol=0.0,
    )
    return target, tuple(holdings), prices, rng.choice([0.0, 0.0025, 0.01])


@pytest.mark.parametrize("seed", range(300))
def test_invariants(seed: int) -> None:
    target, holdings, prices, band = _book(seed)
    plan = generate_orders(
        target, VERDICT, holdings, prices, NAV, params=OrderParams(no_trade_band=band)
    )
    weights = {position.ticker: position.weight for position in target.positions}
    held = {holding.ticker: holding.shares for holding in holdings}
    position = dict(held)
    for order in plan.orders:
        assert isinstance(order.shares, int)
        assert order.shares > 0
        before = position.get(order.ticker, 0)
        after = before + SIGN[order.action] * order.shares
        assert before * after >= 0, f"{order} crosses zero"
        position[order.ticker] = after
    is_reducing = [order.action in REDUCING for order in plan.orders]
    assert is_reducing == sorted(is_reducing, reverse=True)
    assert list(plan.residual_weight) == sorted(weights.keys() | held.keys())
    for ticker, residual in plan.residual_weight.items():
        price, target_w = prices[ticker], weights.get(ticker, 0.0)
        final = position.get(ticker, 0)
        reason = plan.skipped.get(ticker)
        if reason is None:
            assert final == (0 if target_w == 0 else math.trunc(target_w * NAV / price))
        else:
            assert final == held.get(ticker, 0)
        assert residual == pytest.approx(final * price / NAV - target_w, abs=TOL)
        bound = band if reason == SKIPPED_BAND else price / NAV
        assert abs(residual) <= bound + TOL
