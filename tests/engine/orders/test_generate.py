"""Order generation: validation, veto, sizing, legs, ordering."""

from __future__ import annotations

import dataclasses
import math
from datetime import date

import pytest

from hedge_fund.domain.orders import (
    BLOCKED_HALT,
    BLOCKED_VETO,
    SKIPPED_BAND,
    SKIPPED_ROUNDS_TO_ZERO,
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


@pytest.mark.parametrize(
    ("weights", "held", "prices", "message"),
    [
        ({"A": math.nan}, {}, {"A": PRICE}, "weight must be a finite number"),
        ({}, {"A": 1.0}, {"A": PRICE}, "shares must be an int"),
        ({"A": 0.125}, {}, {"A": math.nan}, "price for A must be a finite number > 0"),
    ],
    ids=["weight", "shares", "price"],
)
def test_every_check_runs_before_veto(
    weights: dict[str, float], held: dict[str, int], prices: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _run(weights, held, prices=prices, verdict=_verdict(approved=False, halt=True))


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


# --- legs -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("weights", "held", "expected"),
    [
        pytest.param({"A": 0.125}, {"A": 625}, [_order("A", BUY, 625)], id="long-larger"),
        pytest.param({"A": 0.0625}, {"A": 1250}, [_order("A", SELL, 625)], id="long-smaller"),
        pytest.param({}, {"A": 1250}, [_order("A", SELL, 1250)], id="long-flat"),
        pytest.param({"A": -0.125}, {"A": -625}, [_order("A", SELL_SHORT, 625)], id="short-larger"),
        pytest.param(
            {"A": -0.0625}, {"A": -1250}, [_order("A", BUY_TO_COVER, 625)], id="short-smaller"
        ),
        pytest.param({}, {"A": -1250}, [_order("A", BUY_TO_COVER, 1250)], id="short-flat"),
        pytest.param(
            {"A": -0.0625},
            {"A": 1250},
            [_order("A", SELL, 1250), _order("A", SELL_SHORT, 625)],
            id="long-to-short",
        ),
        pytest.param(
            {"A": 0.0625},
            {"A": -1250},
            [_order("A", BUY_TO_COVER, 1250), _order("A", BUY, 625)],
            id="short-to-long",
        ),
        pytest.param({"A": 0.125}, {}, [_order("A", BUY, 1250)], id="flat-long"),
        pytest.param({"A": -0.125}, {}, [_order("A", SELL_SHORT, 1250)], id="flat-short"),
    ],
)
def test_legs(weights: dict[str, float], held: dict[str, int], expected: list[Order]) -> None:
    plan = _run(weights, held)
    assert plan.orders == tuple(expected)
    assert plan.skipped == {}
    assert plan.blocked is None


def test_order_notional_uses_ref_price() -> None:
    plan = _run({"A": 0.125}, {}, prices={"A": 80.0})
    assert plan.orders == (
        Order(ticker="A", action=BUY, shares=1562, ref_price=80.0, notional=124_960.0),
    )


def test_ignores_extra_prices() -> None:
    plan = _run({"A": 0.125}, {}, prices={"A": PRICE, "Z": math.nan})
    assert plan.orders == (_order("A", BUY, 1250),)
    assert list(plan.residual_weight) == ["A"]


# --- classification ---------------------------------------------------------------------


def test_rounds_toward_zero_for_both_sides() -> None:
    plan = _run({"L": 0.00109, "S": -0.00109}, {})  # 10.9 shares each way
    assert plan.orders == (_order("L", BUY, 10), _order("S", SELL_SHORT, 10))
    assert plan.residual_weight["L"] == pytest.approx(-0.00009)
    assert plan.residual_weight["S"] == pytest.approx(0.00009)


def test_band_skip_keeps_current_shares() -> None:
    plan = _run({"A": 0.1274}, {"A": 1250})
    assert plan.orders == ()
    assert plan.skipped == {"A": SKIPPED_BAND}
    assert plan.residual_weight["A"] == pytest.approx(0.125 - 0.1274)


def test_drift_exactly_at_band_trades() -> None:
    plan = _run({"A": 0.12890625}, {"A": 1250}, band=0.00390625)  # drift == band exactly
    assert plan.orders == (_order("A", BUY, 39),)
    assert plan.skipped == {}


def test_zero_band_never_band_skips() -> None:
    weights, held = {"A": 0.1259765625}, {"A": 1250}  # drift 2**-10, under the default band
    assert _run(weights, held).skipped == {"A": SKIPPED_BAND}
    assert _run(weights, held, band=0.0).orders == (_order("A", BUY, 9),)


def test_rebalance_that_rounds_to_current_is_skipped() -> None:
    plan = _run({"A": 0.125 + 2**-17}, {"A": 1250}, band=0.0)  # 1250.08 shares
    assert plan.orders == ()
    assert plan.skipped == {"A": SKIPPED_ROUNDS_TO_ZERO}


def test_entry_that_rounds_to_zero_is_skipped() -> None:
    # 0.3 shares and below the band: also kills the "entry classified after band" mutant.
    plan = _run({"A": 2**-15}, {})
    assert plan.orders == ()
    assert plan.skipped == {"A": SKIPPED_ROUNDS_TO_ZERO}
    assert plan.residual_weight == {"A": -(2**-15)}


def test_exit_ignores_band() -> None:
    plan = _run({}, {"A": 1})  # 1e-4 of NAV, far below the band
    assert plan.orders == (_order("A", SELL, 1),)


def test_held_excluded_name_exits() -> None:
    plan = _run({"A": 0.125}, {"A": 1250, "B": 625}, excluded={"B": "history"})
    assert plan.orders == (_order("B", SELL, 625),)
    assert plan.skipped == {"A": SKIPPED_BAND}


def test_held_zero_weight_position_exits() -> None:
    assert _run({"A": 0.0}, {"A": 1250}).orders == (_order("A", SELL, 1250),)


@pytest.mark.parametrize(
    ("weights", "held"),
    [({"A": 0.0}, {}), ({}, {"A": 0})],
    ids=["unheld-zero-weight", "zero-share-holding"],
)
def test_silent_ticker(weights: dict[str, float], held: dict[str, int]) -> None:
    plan = _run(weights, held)
    assert plan == OrderPlan(
        as_of=AS_OF, orders=(), skipped={}, residual_weight={"A": 0.0}, blocked=None
    )


def test_band_beats_side_flip() -> None:
    plan = _run({"A": -0.001}, {"A": 10})  # +0.001 held against a -0.001 target
    assert plan.orders == ()
    assert plan.skipped == {"A": SKIPPED_BAND}
    assert plan.residual_weight["A"] == pytest.approx(0.002)


def test_residual_uses_final_shares() -> None:
    assert _run({"A": 0.0625}, {"A": 1250}).residual_weight == {"A": 0.0}
