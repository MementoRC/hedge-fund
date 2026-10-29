"""trades (opening entries and fills on the calendar) and holdings (signed shares per close)."""

from __future__ import annotations

import dataclasses
from datetime import UTC, date, datetime

import pytest

from hedge_fund.domain.ledger import Fill, FillStatus, OpeningEntry
from hedge_fund.domain.orders import OrderAction
from hedge_fund.engine.attribution.book import Trade, holdings, trades

BUY, SELL = OrderAction.BUY, OrderAction.SELL
SHORT, COVER = OrderAction.SELL_SHORT, OrderAction.BUY_TO_COVER
CALENDAR = [
    date(2026, 9, 28),
    date(2026, 9, 29),
    date(2026, 9, 30),
    date(2026, 10, 1),
    date(2026, 10, 2),
    date(2026, 10, 5),
    date(2026, 10, 6),
]
TUESDAY_OPEN = datetime(2026, 9, 29, 13, 30, tzinfo=UTC)  # 09:30 EDT


def _fill(
    cid: str, ticker: str, action: OrderAction, shares: int, price: float, at: datetime
) -> Fill:
    return Fill(
        client_order_id=cid,
        broker_order_id=f"b-{cid}",
        cycle_as_of=date(2026, 9, 25),
        ticker=ticker,
        action=action,
        requested_shares=shares,
        filled_shares=shares,
        avg_price=price,
        filled_at=at,
        status=FillStatus.FILLED,
        recorded_at=at,
    )


def test_fills_are_dated_in_new_york_and_rolled_forward() -> None:
    fills = (
        _fill("a", "AAA", BUY, 10, 100.0, TUESDAY_OPEN),
        _fill("b", "AAA", SELL, 4, 101.0, datetime(2026, 10, 2, 1, 30, tzinfo=UTC)),  # Thu 21:30
        _fill("c", "BBB", SHORT, 3, 50.0, datetime(2026, 10, 3, 15, 0, tzinfo=UTC)),  # Saturday
        _fill("d", "brk.b", COVER, 2, 400.0, datetime(2026, 10, 6, 14, 0, tzinfo=UTC)),
    )
    assert trades((), fills, CALENDAR) == [
        Trade("AAA", date(2026, 9, 29), 10, 100.0),
        Trade("AAA", date(2026, 10, 1), -4, 101.0),
        Trade("BBB", date(2026, 10, 5), -3, 50.0),
        Trade("BRK-B", date(2026, 10, 6), 2, 400.0),
    ]


def test_opening_entries_come_first_at_their_price_and_rolled_date() -> None:
    opening = (
        OpeningEntry("AAA", 10, 98.0, date(2026, 9, 1)),  # before the calendar: its first day
        OpeningEntry("BBB", -5, 49.0, date(2026, 10, 3)),  # Saturday -> Monday
    )
    fill = _fill("a", "AAA", BUY, 1, 100.0, TUESDAY_OPEN)
    assert trades(opening, (fill,), CALENDAR) == [
        Trade("AAA", date(2026, 9, 28), 10, 98.0),
        Trade("BBB", date(2026, 10, 5), -5, 49.0),
        Trade("AAA", date(2026, 9, 29), 1, 100.0),
    ]


def test_zero_fills_and_dates_after_the_calendar_are_left_out() -> None:
    zero = dataclasses.replace(
        _fill("z", "AAA", BUY, 10, 100.0, TUESDAY_OPEN),
        filled_shares=0,
        avg_price=None,
        filled_at=None,
        status=FillStatus.CANCELED,
    )
    late = _fill("late", "AAA", BUY, 1, 100.0, datetime(2026, 10, 7, 14, 0, tzinfo=UTC))
    late_opening = OpeningEntry("CCC", 1, 10.0, date(2026, 10, 7))
    assert trades((late_opening,), (zero, late), CALENDAR) == []


def test_a_counted_fill_without_a_price_raises() -> None:
    broken = dataclasses.replace(_fill("b1", "AAA", BUY, 10, 100.0, TUESDAY_OPEN), avg_price=None)
    with pytest.raises(ValueError, match="'b1' has 10 shares but no avg_price or filled_at"):
        trades((), (broken,), CALENDAR)


def test_holdings_at_each_close() -> None:
    opening = (OpeningEntry("AAA", 10, 98.0, date(2026, 9, 28)),)
    fills = (
        _fill("b1", "AAA", BUY, 5, 100.0, datetime(2026, 9, 30, 13, 30, tzinfo=UTC)),
        _fill("s1", "AAA", SELL, 15, 101.0, datetime(2026, 10, 1, 14, 0, tzinfo=UTC)),
        _fill("x1", "BBB", SHORT, 3, 50.0, datetime(2026, 10, 3, 15, 0, tzinfo=UTC)),
    )
    assert holdings(opening, fills, CALENDAR) == {
        date(2026, 9, 28): {"AAA": 10},
        date(2026, 9, 29): {"AAA": 10},
        date(2026, 9, 30): {"AAA": 15},
        date(2026, 10, 1): {},
        date(2026, 10, 2): {},
        date(2026, 10, 5): {"BBB": -3},
        date(2026, 10, 6): {"BBB": -3},
    }


def test_holdings_net_several_fills_in_a_day_whatever_their_order() -> None:
    fills = [
        _fill("b1", "AAA", BUY, 10, 100.0, TUESDAY_OPEN),
        _fill("s1", "AAA", SELL, 15, 101.0, datetime(2026, 9, 29, 15, 0, tzinfo=UTC)),
        _fill("b2", "AAA", BUY, 2, 102.0, datetime(2026, 9, 29, 19, 0, tzinfo=UTC)),
    ]
    book = holdings((), fills, CALENDAR)
    assert book[date(2026, 9, 28)] == {}
    assert book[date(2026, 9, 29)] == {"AAA": -3}
    assert holdings((), fills[::-1], CALENDAR) == book
