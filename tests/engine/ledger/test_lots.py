"""build_lots: signed FIFO lots from the opening balance and the fills."""

from __future__ import annotations

import dataclasses
from datetime import UTC, date, datetime, timedelta

import pytest

from hedge_fund.domain.ledger import ClosedLot, Fill, FillStatus, Lot, OpeningEntry
from hedge_fund.domain.orders import OrderAction
from hedge_fund.engine.ledger.lots import OPENING_SOURCE, build_lots

BUY, SELL = OrderAction.BUY, OrderAction.SELL
SHORT, COVER = OrderAction.SELL_SHORT, OrderAction.BUY_TO_COVER
T0 = datetime(2026, 10, 1, 13, 30, tzinfo=UTC)  # 09:30 EDT
OPENED = date(2026, 9, 1)
MIDNIGHT = datetime(2026, 9, 1, tzinfo=UTC)


def _at(minutes: int) -> datetime:
    return T0 + timedelta(minutes=minutes)


def _fill(
    cid: str, ticker: str, action: OrderAction, shares: int, price: float, at: datetime = T0
) -> Fill:
    return Fill(
        client_order_id=cid,
        broker_order_id=f"b-{cid}",
        cycle_as_of=date(2026, 9, 30),
        ticker=ticker,
        action=action,
        requested_shares=shares,
        filled_shares=shares,
        avg_price=price,
        filled_at=at,
        status=FillStatus.FILLED,
        recorded_at=_at(60),
    )


def _zero(cid: str, status: FillStatus) -> Fill:
    return Fill(
        client_order_id=cid,
        broker_order_id=None,
        cycle_as_of=date(2026, 9, 30),
        ticker="AAA",
        action=BUY,
        requested_shares=10,
        filled_shares=0,
        avg_price=None,
        filled_at=None,
        status=status,
        recorded_at=_at(60),
    )


def _opening(ticker: str, shares: int, price: float, on: date = OPENED) -> OpeningEntry:
    return OpeningEntry(ticker=ticker, shares=shares, price=price, on=on)


def _closed(
    lot: Lot, shares: int, exit_price: float, exited_at: datetime, exit_cid: str
) -> ClosedLot:
    return ClosedLot(
        ticker=lot.ticker,
        shares=shares,
        entry_price=lot.entry_price,
        entered_at=lot.entered_at,
        source=lot.source,
        exit_price=exit_price,
        exited_at=exited_at,
        exit_client_order_id=exit_cid,
        realised_pnl=(exit_price - lot.entry_price) * shares,
    )


def test_nothing_gives_nothing() -> None:
    assert build_lots((), ()) == ((), ())


def test_opening_entries_seed_lots_sorted_by_ticker() -> None:
    lots, closed = build_lots((_opening("BBB", -5, 50.0), _opening("AAA", 10, 90.0)), ())
    assert lots == (
        Lot(ticker="AAA", shares=10, entry_price=90.0, entered_at=MIDNIGHT, source=OPENING_SOURCE),
        Lot(ticker="BBB", shares=-5, entry_price=50.0, entered_at=MIDNIGHT, source=OPENING_SOURCE),
    )
    assert closed == ()
    assert OPENING_SOURCE == "opening"


def test_fifo_across_several_lots_with_a_partial_close() -> None:
    first = _fill("b1", "AAA", BUY, 10, 100.0, _at(0))
    second = _fill("b2", "AAA", BUY, 10, 110.0, _at(1))
    sell = _fill("s1", "AAA", SELL, 15, 120.0, _at(2))
    lots, closed = build_lots((), (first, second, sell))
    lot1 = Lot("AAA", 10, 100.0, _at(0), "b1")
    lot2 = Lot("AAA", 10, 110.0, _at(1), "b2")
    assert closed == (_closed(lot1, 10, 120.0, _at(2), "s1"), _closed(lot2, 5, 120.0, _at(2), "s1"))
    assert [c.realised_pnl for c in closed] == [200.0, 50.0]
    assert lots == (Lot("AAA", 5, 110.0, _at(1), "b2"),)


def test_a_sell_that_crosses_zero_closes_the_long_then_opens_a_short() -> None:
    lots, closed = build_lots(
        (), (_fill("b1", "AAA", BUY, 10, 100.0, _at(0)), _fill("s1", "AAA", SELL, 15, 90.0, _at(1)))
    )
    assert closed == (_closed(Lot("AAA", 10, 100.0, _at(0), "b1"), 10, 90.0, _at(1), "s1"),)
    assert closed[0].realised_pnl == -100.0
    assert lots == (Lot("AAA", -5, 90.0, _at(1), "s1"),)


def test_a_buy_that_crosses_zero_covers_the_short_then_opens_a_long() -> None:
    lots, closed = build_lots(
        (),
        (_fill("x1", "AAA", SHORT, 10, 100.0, _at(0)), _fill("b1", "AAA", BUY, 15, 80.0, _at(1))),
    )
    assert closed == (_closed(Lot("AAA", -10, 100.0, _at(0), "x1"), -10, 80.0, _at(1), "b1"),)
    assert closed[0].realised_pnl == 200.0
    assert lots == (Lot("AAA", 5, 80.0, _at(1), "b1"),)


@pytest.mark.parametrize(("cover_price", "pnl"), [(40.0, 40.0), (60.0, -40.0)])
def test_short_realised_pnl_sign(cover_price: float, pnl: float) -> None:
    lots, closed = build_lots(
        (),
        (
            _fill("x1", "AAA", SHORT, 10, 50.0, _at(0)),
            _fill("c1", "AAA", COVER, 4, cover_price, _at(1)),
        ),
    )
    assert [(c.shares, c.realised_pnl) for c in closed] == [(-4, pnl)]
    assert lots == (Lot("AAA", -6, 50.0, _at(0), "x1"),)


def test_fills_apply_by_filled_at_not_by_id() -> None:
    sell = _fill("a-sell", "AAA", SELL, 5, 120.0, _at(1))
    buy = _fill("z-buy", "AAA", BUY, 10, 100.0, _at(0))
    lots, closed = build_lots((), (sell, buy))
    assert [(c.source, c.shares, c.realised_pnl) for c in closed] == [("z-buy", 5, 100.0)]
    assert lots == (Lot("AAA", 5, 100.0, _at(0), "z-buy"),)


def test_at_one_instant_a_cover_applies_before_a_buy_on_a_short() -> None:
    buy = _fill("a-buy", "AAA", BUY, 5, 90.0)  # sorts first by id
    cover = _fill("b-cover", "AAA", COVER, 10, 90.0)
    lots, closed = build_lots((_opening("AAA", -10, 100.0),), (buy, cover))
    assert [(c.exit_client_order_id, c.shares) for c in closed] == [("b-cover", -10)]
    assert lots == (Lot("AAA", 5, 90.0, T0, "a-buy"),)


def test_at_one_instant_ids_break_the_tie() -> None:
    later_id = _fill("b-buy", "AAA", BUY, 5, 101.0)
    earlier_id = _fill("a-buy", "AAA", BUY, 5, 100.0)
    sell = _fill("s1", "AAA", SELL, 5, 110.0, _at(1))
    lots, closed = build_lots((), (later_id, sell, earlier_id))
    assert [c.source for c in closed] == ["a-buy"]
    assert lots == (Lot("AAA", 5, 101.0, T0, "b-buy"),)


def test_zero_share_fills_create_no_lots() -> None:
    zeros = (_zero("c1", FillStatus.CANCELED), _zero("n1", FillStatus.NOT_SUBMITTED))
    assert build_lots((), zeros) == ((), ())


@pytest.mark.parametrize(
    ("on", "filled_at", "raises"),
    [
        pytest.param(date(2026, 10, 1), T0, True, id="same-day"),
        pytest.param(date(2026, 10, 2), T0, True, id="after"),
        pytest.param(date(2026, 9, 30), T0, False, id="before"),
        pytest.param(
            date(2026, 9, 30), datetime(2026, 10, 1, 2, 0, tzinfo=UTC), True, id="new-york-date"
        ),
    ],
)
def test_opening_must_predate_the_first_fills_new_york_date(
    on: date, filled_at: datetime, raises: bool
) -> None:
    opening = (_opening("AAA", 10, 90.0, on),)
    fills = (
        _fill("late", "AAA", BUY, 1, 100.0, datetime(2026, 10, 5, 14, 0, tzinfo=UTC)),
        _fill("early", "BBB", BUY, 1, 100.0, filled_at),  # another ticker sets the date
    )
    if raises:
        with pytest.raises(ValueError, match="opening entries must be dated before .*: AAA"):
            build_lots(opening, fills)
    else:
        lots, _ = build_lots(opening, fills)
        assert [(lot.ticker, lot.shares, lot.source) for lot in lots] == [
            ("AAA", 10, OPENING_SOURCE),
            ("AAA", 1, "late"),
            ("BBB", 1, "early"),
        ]


def test_without_a_non_zero_fill_any_opening_date_is_accepted() -> None:
    opening = (_opening("AAA", 10, 90.0, date(2026, 12, 31)),)
    lots, _ = build_lots(opening, (_zero("c1", FillStatus.CANCELED),))
    assert [lot.source for lot in lots] == [OPENING_SOURCE]


def test_a_non_zero_fill_without_price_or_time_raises() -> None:
    broken = dataclasses.replace(_fill("b1", "AAA", BUY, 10, 100.0), filled_at=None)
    with pytest.raises(ValueError, match="'b1' has 10 shares but no avg_price or filled_at"):
        build_lots((), (broken,))


def test_inputs_are_not_mutated() -> None:
    opening = [_opening("AAA", 10, 90.0)]
    fills = [_fill("s1", "AAA", SELL, 4, 95.0), _fill("s2", "AAA", SELL, 2, 96.0, _at(1))]
    before = (list(opening), list(fills))
    first = build_lots(opening, fills)
    assert (opening, fills) == before
    assert build_lots(opening, fills) == first
