"""mark: open lots at their ticker's last usable close, or unmarked; never raises."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime

import pandas as pd

from hedge_fund.domain.ledger import Lot, MarkedLot
from hedge_fund.engine.ledger.mark import mark

AS_OF = date(2026, 10, 1)
NAN = math.nan
AT = datetime(2026, 10, 1, 13, 30, tzinfo=UTC)
INDEX = pd.to_datetime(["2026-09-23", "2026-09-24", "2026-09-30", "2026-10-01", "2026-10-02"])
CLOSES = pd.DataFrame(
    {
        "AAA": [1.0, 2.0, 3.0, 14.0, 99.0],  # 14.0 on as_of; 10-02 is after as_of
        "BBB": [20.0, 21.0, NAN, NAN, NAN],  # last close 09-24: exactly 7 days old
        "CCC": [30.0, NAN, NAN, NAN, NAN],  # last close 09-23: 8 days old
        "DDD": [40.0, 41.0, 42.0, 0.0, -1.0],  # a zero close is skipped: 42.0 on 09-30
        "EEE": [NAN, NAN, 50.0, math.inf, NAN],  # inf is not a close: 50.0 on 09-30
    },
    index=INDEX,
)


def _lot(ticker: str, shares: int, entry_price: float) -> Lot:
    return Lot(ticker=ticker, shares=shares, entry_price=entry_price, entered_at=AT, source="x")


LOTS = (
    _lot("AAA", 10, 10.0),
    _lot("BBB", -5, 25.0),
    _lot("CCC", 3, 30.0),
    _lot("DDD", 2, 40.0),
    _lot("EEE", 1, 50.0),
    _lot("FFF", 4, 1.0),  # no column
)


def test_marks_and_unmarked() -> None:
    marked, unmarked = mark(LOTS, CLOSES, as_of=AS_OF)
    assert marked == (
        MarkedLot(lot=LOTS[0], price=14.0, close_date=date(2026, 10, 1), unrealised_pnl=40.0),
        MarkedLot(lot=LOTS[1], price=21.0, close_date=date(2026, 9, 24), unrealised_pnl=20.0),
        MarkedLot(lot=LOTS[3], price=42.0, close_date=date(2026, 9, 30), unrealised_pnl=4.0),
        MarkedLot(lot=LOTS[4], price=50.0, close_date=date(2026, 9, 30), unrealised_pnl=0.0),
    )
    assert unmarked == (LOTS[2], LOTS[5])


def test_every_lot_of_a_ticker_gets_the_same_close() -> None:
    lots = (_lot("AAA", 10, 10.0), _lot("AAA", 5, 12.0))
    marked, _ = mark(lots, CLOSES, as_of=AS_OF)
    assert [(m.price, m.unrealised_pnl) for m in marked] == [(14.0, 40.0), (14.0, 10.0)]


def test_max_age_days_eight_marks_an_eight_day_old_close() -> None:
    marked, unmarked = mark((LOTS[2],), CLOSES, as_of=AS_OF, max_age_days=8)
    assert marked == (MarkedLot(LOTS[2], 30.0, date(2026, 9, 23), 0.0),)
    assert unmarked == ()


def test_max_age_days_zero_needs_a_close_on_as_of() -> None:
    marked, unmarked = mark(LOTS[:2], CLOSES, as_of=AS_OF, max_age_days=0)
    assert [m.lot.ticker for m in marked] == ["AAA"]
    assert unmarked == (LOTS[1],)


def test_no_lots_gives_nothing() -> None:
    assert mark((), CLOSES, as_of=AS_OF) == ((), ())


def test_prices_are_floats() -> None:
    marked, _ = mark(LOTS[:1], CLOSES, as_of=AS_OF)
    assert type(marked[0].price) is float


def test_the_frame_is_not_mutated() -> None:
    before = CLOSES.copy()
    mark(LOTS, CLOSES, as_of=AS_OF)
    pd.testing.assert_frame_equal(CLOSES, before)
