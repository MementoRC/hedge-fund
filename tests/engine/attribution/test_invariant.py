"""The attribution invariant against the 7a ledger, and the regressed split against a real
day_model."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime
from typing import cast

import pandas as pd
import pytest

from hedge_fund.domain.ledger import Fill, FillStatus, OpeningEntry
from hedge_fund.domain.orders import OrderAction
from hedge_fund.engine.attribution import (
    DayModel,
    Trade,
    attribute_day,
    day_model,
    holdings,
    trades,
    trading_date,
)
from hedge_fund.engine.execution.window import MARKET_TZ
from hedge_fund.engine.ledger import build_lots, mark, signed_quantity
from hedge_fund.engine.portfolio.risk_model import simple_returns
from tests.engine.attribution.test_model import (
    BENCHMARK,
    DAY,
    PARAMS,
    RETURNS,
    SECTORS,
    UNIVERSE,
)

BUY, SELL = OrderAction.BUY, OrderAction.SELL
SHORT, COVER = OrderAction.SELL_SHORT, OrderAction.BUY_TO_COVER
DAYS = pd.bdate_range("2026-09-21", "2026-10-09")  # 15 weekdays, gap-free closes
CALENDAR = [stamp.date() for stamp in DAYS]
CLOSES = pd.DataFrame(
    {
        "AAA": [100, 101, 99.5, 102, 103, 104.5, 103, 105, 106, 104, 107, 108, 107.5, 109, 110],
        "BBB": [50, 51, 52, 50.5, 49, 48, 49.5, 51, 50, 52, 53, 52.5, 54, 53, 55],
    },
    index=DAYS,
    dtype=float,
)
MODEL = DayModel(
    market=0.004,
    sector_returns={"Tech": -0.001},
    betas={"AAA": 1.2},
    sectors={"AAA": "Tech"},
    residuals={"AAA": 0.0},
)


def _fill(
    cid: str, ticker: str, action: OrderAction, shares: int, price: float, at: datetime
) -> Fill:
    return Fill(
        client_order_id=cid,
        broker_order_id=f"b-{cid}",
        cycle_as_of=date(2026, 9, 24),
        ticker=ticker,
        action=action,
        requested_shares=shares,
        filled_shares=shares,
        avg_price=price,
        filled_at=at,
        status=FillStatus.FILLED,
        recorded_at=at,
    )


OPENING = (
    OpeningEntry("AAA", 10, 98.0, date(2026, 9, 18)),
    OpeningEntry("BBB", -20, 49.0, date(2026, 9, 24)),
)
ZERO = Fill(
    client_order_id="z1",
    broker_order_id=None,
    cycle_as_of=date(2026, 9, 24),
    ticker="AAA",
    action=BUY,
    requested_shares=4,
    filled_shares=0,
    avg_price=None,
    filled_at=None,
    status=FillStatus.CANCELED,
    recorded_at=datetime(2026, 9, 25, 20, 0, tzinfo=UTC),
)
FILLS = (
    _fill("f1", "AAA", BUY, 5, 103.2, datetime(2026, 9, 25, 14, 0, tzinfo=UTC)),
    _fill("f2", "AAA", SELL, 20, 104.0, datetime(2026, 9, 30, 15, 0, tzinfo=UTC)),
    _fill("f3", "AAA", COVER, 2, 104.5, datetime(2026, 9, 30, 18, 0, tzinfo=UTC)),
    _fill("f6", "BBB", BUY, 5, 50.2, datetime(2026, 10, 2, 1, 0, tzinfo=UTC)),
    _fill("f4", "BBB", SHORT, 10, 51.5, datetime(2026, 10, 3, 15, 0, tzinfo=UTC)),
    _fill("f5", "AAA", BUY, 3, 109.5, datetime(2026, 10, 9, 14, 0, tzinfo=UTC)),
    ZERO,
)


def _attributed(start: date, end: date) -> dict[str, float]:
    """Per ticker, every component of every row over the window's trading days."""
    book = holdings(OPENING, FILLS, CALENDAR)
    dated = trades(OPENING, FILLS, CALENDAR)
    parts: dict[str, list[float]] = {}
    for position, day in enumerate(CALENDAR):
        if not start <= day <= end:
            continue
        model = MODEL if position % 2 == 0 else None
        today = [trade for trade in dated if trade.day == day]
        held = book[CALENDAR[position - 1]]
        rows, excluded = attribute_day(day, held, today, CLOSES, model, {"AAA": "Tech"})
        assert excluded == {}
        for row in rows:
            parts.setdefault(row.ticker, []).extend(
                (row.market, row.sector_pnl, row.residual, row.trading)
            )
    return {ticker: math.fsum(values) for ticker, values in parts.items()}


def _rolled(on: date) -> date:
    return cast(date, trading_date(on, CALENDAR))


def _ledger_pnl(last: date) -> dict[str, float]:
    """Realised + unrealised per ticker over the entries and fills rolled to <= ``last``,
    marked at ``last``'s close."""
    opening = [entry for entry in OPENING if _rolled(entry.on) <= last]
    fills = [
        fill
        for fill in FILLS
        if signed_quantity(fill)
        and _rolled(cast(datetime, fill.filled_at).astimezone(MARKET_TZ).date()) <= last
    ]
    lots, closed = build_lots(opening, fills)
    marked, unmarked = mark(lots, CLOSES, as_of=last)
    assert unmarked == ()
    parts: dict[str, list[float]] = {}
    for done in closed:
        parts.setdefault(done.ticker, []).append(done.realised_pnl)
    for held in marked:
        parts.setdefault(held.lot.ticker, []).append(held.unrealised_pnl)
    return {ticker: math.fsum(values) for ticker, values in parts.items()}


def test_the_fixture_dates_roll_forward_as_intended() -> None:
    assert trades(OPENING, FILLS, CALENDAR) == [
        Trade("AAA", date(2026, 9, 21), 10, 98.0),
        Trade("BBB", date(2026, 9, 24), -20, 49.0),
        Trade("AAA", date(2026, 9, 25), 5, 103.2),
        Trade("AAA", date(2026, 9, 30), -20, 104.0),
        Trade("AAA", date(2026, 9, 30), 2, 104.5),
        Trade("BBB", date(2026, 10, 1), 5, 50.2),
        Trade("BBB", date(2026, 10, 5), -10, 51.5),
        Trade("AAA", date(2026, 10, 9), 3, 109.5),
    ]


@pytest.mark.parametrize(
    ("start", "end"),
    [
        pytest.param(date(2026, 9, 22), date(2026, 10, 9), id="whole-history"),
        pytest.param(date(2026, 9, 24), date(2026, 10, 2), id="opening-entry-in-the-window"),
        pytest.param(date(2026, 9, 30), date(2026, 10, 8), id="mid-history"),
        pytest.param(date(2026, 10, 5), date(2026, 10, 5), id="weekend-fill-day"),
        pytest.param(date(2026, 9, 25), date(2026, 9, 25), id="baseline-fill-day"),
    ],
)
def test_attributed_pnl_is_the_change_in_ledger_pnl(start: date, end: date) -> None:
    previous = CALENDAR[CALENDAR.index(start) - 1]
    after, before = _ledger_pnl(end), _ledger_pnl(previous)
    attributed = _attributed(start, end)
    assert attributed  # the window has rows
    for ticker in sorted(after.keys() | before.keys() | attributed.keys()):
        expected = after.get(ticker, 0.0) - before.get(ticker, 0.0)
        assert attributed.get(ticker, 0.0) == pytest.approx(expected, abs=1e-6), ticker


def test_a_regressed_holding_splits_as_the_day_regression_does() -> None:
    returns = RETURNS[["A1", "A2", "B1", "B2"]].copy()
    returns.loc[returns.index[-1], "B2"] = 0.02  # off the exact fit: non-zero residuals
    closes = 100.0 * (1.0 + returns).cumprod()
    model = day_model(simple_returns(closes), BENCHMARK, SECTORS, UNIVERSE, PARAMS, DAY)
    assert model is not None
    assert max(abs(u) for u in model.residuals.values()) > 1e-4
    rows, excluded = attribute_day(DAY, {"A1": 10, "B2": -7}, (), closes, model, SECTORS)
    assert excluded == {}
    previous = closes.iloc[-2]
    assert [row.ticker for row in rows] == ["A1", "B2"]
    for row in rows:
        exposure = row.prior_shares * float(previous[row.ticker])
        sector = model.sectors[row.ticker]
        assert row.regressed
        assert row.market == pytest.approx(exposure * model.betas[row.ticker] * model.market)
        assert row.sector_pnl == pytest.approx(exposure * model.sector_returns[sector])
        assert row.residual == pytest.approx(exposure * model.residuals[row.ticker], abs=1e-9)
