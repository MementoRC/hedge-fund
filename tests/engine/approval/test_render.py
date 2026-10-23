"""Approval summary: golden text for every block of the layout."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from datetime import date

import pytest

from hedge_fund.domain.orders import (
    BLOCKED_HALT,
    BLOCKED_VETO,
    SKIPPED_BAND,
    SKIPPED_ROUNDS_TO_ZERO,
    Order,
    OrderAction,
    OrderPlan,
)
from hedge_fund.domain.risk import BreakerState, RiskBreakdown, RiskVerdict, Violation
from hedge_fund.engine.approval.render import render_proposal

AS_OF = date(2026, 10, 23)
BUY = OrderAction.BUY
SELL = OrderAction.SELL
SELL_SHORT = OrderAction.SELL_SHORT
BUY_TO_COVER = OrderAction.BUY_TO_COVER
QUIET = RiskBreakdown(
    total_vol=0.05,
    factor_vol=0.0,
    specific_vol=0.05,
    factor_share=0.0,
    by_factor={},
    by_name={},
    unassessed=(),
)
CALM = BreakerState(daily_loss=0.0, drawdown=0.0, daily_tripped=False, drawdown_tripped=False)


def _order(ticker: str, action: OrderAction, shares: int, ref_price: float) -> Order:
    return Order(
        ticker=ticker,
        action=action,
        shares=shares,
        ref_price=ref_price,
        notional=shares * ref_price,
    )


def _plan(
    *orders: Order,
    skipped: Mapping[str, str] | None = None,
    residuals: Mapping[str, float] | None = None,
    blocked: str | None = None,
) -> OrderPlan:
    return OrderPlan(
        as_of=AS_OF,
        orders=orders,
        skipped=skipped or {},
        residual_weight=residuals or {},
        blocked=blocked,
    )


def _verdict(
    *,
    breakdown: RiskBreakdown = QUIET,
    breakers: BreakerState = CALM,
    violations: tuple[Violation, ...] = (),
    halt: bool = False,
) -> RiskVerdict:
    return RiskVerdict(
        as_of=AS_OF,
        approved=not violations and not halt,
        halt=halt,
        violations=violations,
        breakdown=breakdown,
        breakers=breakers,
    )


def _blocks(text: str) -> list[str]:
    """Header, orders, skipped + residuals, risk and violations blocks."""
    return text.removesuffix("\n").split("\n\n")


# --- whole summaries --------------------------------------------------------------------

EXAMPLE_VERDICT = _verdict(
    breakdown=RiskBreakdown(
        total_vol=0.118,
        factor_vol=0.1,
        specific_vol=0.0727,
        factor_share=0.62,
        by_factor={"momentum": 0.0038, "quality": 0.0002, "size": -0.0009, "value": 0.0021},
        by_name={
            "AAPL": 0.14,
            "JPM": 0.082,
            "KO": 0.01,
            "MSFT": 0.11,
            "NVDA": -0.031,
            "XOM": 0.095,
        },
        unassessed=("BRK.B",),
    ),
    breakers=BreakerState(
        daily_loss=-0.004, drawdown=0.021, daily_tripped=False, drawdown_tripped=False
    ),
)
EXAMPLE_PLAN = _plan(
    _order("AAPL", SELL, 40, 231.10),
    _order("XOM", BUY_TO_COVER, 25, 118.40),
    _order("MSFT", BUY, 12, 412.05),
    _order("XOM", BUY, 30, 118.40),
    skipped={"JPM": SKIPPED_BAND, "KO": SKIPPED_ROUNDS_TO_ZERO, "NVDA": SKIPPED_BAND},
    residuals={
        "AAPL": 0.0042,
        "JPM": 0.0001,
        "KO": 0.0003,
        "MSFT": -0.0031,
        "NVDA": -0.0008,
        "XOM": 0.0012,
    },
)
EXAMPLE = """\
Proposal for 2026-10-23   risk: APPROVED

Orders (reducing first)
  leg     ticker  action        shares     ref  notional
  reduce  AAPL    sell              40  231.10  9,244.00
  reduce  XOM     buy_to_cover      25  118.40  2,960.00
  open    MSFT    buy               12  412.05  4,944.60
  open    XOM     buy               30  118.40  3,552.00   deferred: side flip
  ---
  4 orders   buys 11,456.60   sells 9,244.00   gross 20,700.60   net +2,212.60

Skipped   JPM band, KO rounds-to-zero, NVDA band
Largest residuals   AAPL +0.42%, MSFT -0.31%, XOM +0.12%, NVDA -0.08%, KO +0.03%

Risk   vol 11.80%  factor share 62.00%  specific 7.27%
  top factors   momentum +38.00%, value +21.00%, size -9.00%
  top names     AAPL +14.00%, MSFT +11.00%, XOM +9.50%, JPM +8.20%, NVDA -3.10%
  unassessed    BRK.B
  breakers      daily -0.40%   drawdown 2.10%

Violations   (none)
"""


def test_spec_example() -> None:
    assert render_proposal(EXAMPLE_VERDICT, EXAMPLE_PLAN) == EXAMPLE


def test_single_order_on_a_quiet_book() -> None:
    plan = _plan(_order("KO", SELL_SHORT, 100, 61.25))
    assert render_proposal(_verdict(), plan) == (
        """\
Proposal for 2026-10-23   risk: APPROVED

Orders (reducing first)
  leg   ticker  action      shares    ref  notional
  open  KO      sell_short     100  61.25  6,125.00
  ---
  1 order   buys 0.00   sells 6,125.00   gross 6,125.00   net -6,125.00

Skipped   (none)
Largest residuals   (none)

Risk   vol 5.00%  factor share 0.00%  specific 5.00%
  top factors   (none)
  top names     (none)
  breakers      daily +0.00%   drawdown 0.00%

Violations   (none)
"""
    )


def test_empty_plan() -> None:
    plan = _plan(skipped={"KO": SKIPPED_BAND}, residuals={"KO": 0.001})
    assert render_proposal(_verdict(), plan) == (
        """\
Proposal for 2026-10-23   risk: APPROVED

Orders   (none)

Skipped   KO band
Largest residuals   KO +0.10%

Risk   vol 5.00%  factor share 0.00%  specific 5.00%
  top factors   (none)
  top names     (none)
  breakers      daily +0.00%   drawdown 0.00%

Violations   (none)
"""
    )


def test_halted_plan() -> None:
    breakers = BreakerState(
        daily_loss=-0.031, drawdown=0.12, daily_tripped=True, drawdown_tripped=True
    )
    verdict = _verdict(breakers=breakers, halt=True)
    plan = _plan(residuals={"AAPL": 0.05, "MSFT": -0.02}, blocked=BLOCKED_HALT)
    assert render_proposal(verdict, plan) == (
        """\
Proposal for 2026-10-23   risk: HALT

Orders   (none)

Skipped   (none)
Largest residuals   AAPL +5.00%, MSFT -2.00%

Risk   vol 5.00%  factor share 0.00%  specific 5.00%
  top factors   (none)
  top names     (none)
  breakers      daily -3.10% TRIPPED   drawdown 12.00% TRIPPED

Violations   (none)
"""
    )


def test_vetoed_plan_with_violations() -> None:
    violations = (
        Violation(check="gross", observed=1.62, limit=1.5, detail="gross exposure above limit"),
        Violation(check="name:AAPL", observed=0.12, limit=0.1, detail=""),
        Violation(
            check="unassessable:ZZZ",
            observed=math.nan,
            limit=math.nan,
            detail="no factor exposures",
        ),
    )
    verdict = _verdict(
        breakdown=dataclasses.replace(QUIET, unassessed=("ZZZ",)), violations=violations
    )
    assert render_proposal(verdict, _plan(blocked=BLOCKED_VETO)) == (
        """\
Proposal for 2026-10-23   risk: VETO

Orders   (none)

Skipped   (none)
Largest residuals   (none)

Risk   vol 5.00%  factor share 0.00%  specific 5.00%
  top factors   (none)
  top names     (none)
  unassessed    ZZZ
  breakers      daily +0.00%   drawdown 0.00%

Violations
  gross  observed 1.6200  limit 1.5000  gross exposure above limit
  name:AAPL  observed 0.1200  limit 0.1000
  unassessable:ZZZ  observed n/a  limit n/a  no factor exposures
"""
    )


# --- orders block -----------------------------------------------------------------------


def test_columns_widen_for_long_tickers_and_large_numbers() -> None:
    plan = _plan(_order("A", SELL, 1, 5.0), _order("BERKSHIRE.B", BUY, 12500, 1234.5))
    assert _blocks(render_proposal(_verdict(), plan))[1] == (
        """\
Orders (reducing first)
  leg     ticker       action  shares       ref       notional
  reduce  A            sell         1      5.00           5.00
  open    BERKSHIRE.B  buy      12500  1,234.50  15,431,250.00
  ---
  2 orders   buys 15,431,250.00   sells 5.00   gross 15,431,255.00   net +15,431,245.00"""
    )


def test_blocked_plan_with_orders_raises() -> None:
    plan = _plan(_order("A", SELL, 1, 5.0), blocked=BLOCKED_VETO)
    with pytest.raises(ValueError, match=r"blocked plan \(veto\) must have no orders"):
        render_proposal(_verdict(violations=(Violation("gross", 2.0, 1.5, ""),)), plan)


def test_unknown_blocked_value_raises() -> None:
    with pytest.raises(ValueError, match="unknown blocked value 'paused'"):
        render_proposal(_verdict(), _plan(blocked="paused"))


def test_package_exports() -> None:
    import hedge_fund.engine.approval as approval

    assert approval.__all__ == ["plan_fingerprint", "render_proposal"]
    assert approval.render_proposal is render_proposal


# --- skipped and residuals --------------------------------------------------------------


def test_residual_ties_go_to_the_smaller_ticker_and_zero_is_positive() -> None:
    plan = _plan(
        skipped={"A": SKIPPED_BAND, "B": SKIPPED_ROUNDS_TO_ZERO},
        residuals={"C": -0.00004, "B": -0.001, "A": 0.001},
    )
    assert _blocks(render_proposal(_verdict(), plan))[2] == (
        "Skipped   A band, B rounds-to-zero\nLargest residuals   A +0.10%, B -0.10%, C +0.00%"
    )


# --- risk block -------------------------------------------------------------------------


def _risk_lines(breakdown: RiskBreakdown = QUIET, breakers: BreakerState = CALM) -> list[str]:
    text = render_proposal(_verdict(breakdown=breakdown, breakers=breakers), _plan())
    return _blocks(text)[3].split("\n")


@pytest.mark.parametrize("factor_vol", [0.0, 1e-200], ids=["zero", "variance-underflows"])
def test_no_top_factors_without_factor_variance(factor_vol: float) -> None:
    breakdown = dataclasses.replace(QUIET, factor_vol=factor_vol, by_factor={"value": 0.001})
    assert _risk_lines(breakdown)[1] == "  top factors   (none)"


def test_unassessed_lists_every_ticker() -> None:
    breakdown = dataclasses.replace(QUIET, unassessed=("BRK.B", "ZZZ"))
    assert _risk_lines(breakdown)[3] == "  unassessed    BRK.B, ZZZ"


@pytest.mark.parametrize(
    ("breakers", "line"),
    [
        pytest.param(
            BreakerState(
                daily_loss=-0.031, drawdown=0.05, daily_tripped=True, drawdown_tripped=False
            ),
            "  breakers      daily -3.10% TRIPPED   drawdown 5.00%",
            id="daily",
        ),
        pytest.param(
            BreakerState(
                daily_loss=0.012, drawdown=0.12, daily_tripped=False, drawdown_tripped=True
            ),
            "  breakers      daily +1.20%   drawdown 12.00% TRIPPED",
            id="drawdown",
        ),
    ],
)
def test_breakers_line(breakers: BreakerState, line: str) -> None:
    assert _risk_lines(breakers=breakers)[-1] == line


def test_non_finite_risk_numbers_print_as_na() -> None:
    breakdown = RiskBreakdown(
        total_vol=math.nan,
        factor_vol=math.nan,
        specific_vol=math.inf,
        factor_share=math.nan,
        by_factor={"value": 0.01},
        by_name={"AAPL": math.nan},
        unassessed=(),
    )
    breakers = dataclasses.replace(CALM, daily_loss=math.nan, drawdown=math.inf)
    assert _risk_lines(breakdown, breakers) == [
        "Risk   vol n/a  factor share n/a  specific n/a",
        "  top factors   value n/a",
        "  top names     AAPL n/a",
        "  breakers      daily n/a   drawdown n/a",
    ]
