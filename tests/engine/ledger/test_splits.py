"""apply_splits: restate opening entries and fills in post-split units."""

from __future__ import annotations

import dataclasses
from datetime import UTC, date, datetime
from fractions import Fraction

import pytest

from hedge_fund.domain.ledger import Fill, FillStatus, OpeningEntry, Split
from hedge_fund.domain.orders import OrderAction
from hedge_fund.engine.ledger.splits import apply_splits, split_window

EX = date(2026, 10, 5)
BEFORE = datetime(2026, 10, 2, 13, 30, tzinfo=UTC)
AFTER = datetime(2026, 10, 5, 13, 30, tzinfo=UTC)


def _fill(
    cid: str = "f1",
    ticker: str = "AAA",
    shares: int = 10,
    price: float = 100.0,
    at: datetime = BEFORE,
) -> Fill:
    return Fill(
        client_order_id=cid,
        broker_order_id=f"b-{cid}",
        cycle_as_of=date(2026, 10, 1),
        ticker=ticker,
        action=OrderAction.BUY,
        requested_shares=shares,
        filled_shares=shares,
        avg_price=price,
        filled_at=at,
        status=FillStatus.FILLED,
        recorded_at=at,
    )


def _opening(ticker: str = "AAA", shares: int = 10, price: float = 100.0) -> OpeningEntry:
    return OpeningEntry(ticker=ticker, shares=shares, price=price, on=date(2026, 9, 1))


def _split(ratio: Fraction, ticker: str = "AAA", ex_date: date = EX) -> Split:
    return Split(ticker=ticker, ex_date=ex_date, ratio=ratio)


def test_a_forward_split_scales_a_fill_before_the_ex_date() -> None:
    fill = _fill()
    _, (out,) = apply_splits((), (fill,), (_split(Fraction(2)),))
    assert (out.filled_shares, out.requested_shares, out.avg_price) == (20, 20, 50.0)
    assert fill.filled_shares == 10
    assert out == dataclasses.replace(fill, filled_shares=20, requested_shares=20, avg_price=50.0)


def test_a_fill_on_or_after_the_ex_date_is_untouched() -> None:
    fill = _fill(at=AFTER)
    assert apply_splits((), (fill,), (_split(Fraction(2)),)) == ((), (fill,))


def test_an_opening_entry_before_the_ex_date_is_scaled() -> None:
    (out,), _ = apply_splits((_opening(),), (), (_split(Fraction(2)),))
    assert (out.shares, out.price) == (20, 50.0)


def test_an_opening_entry_on_the_ex_date_is_untouched() -> None:
    entry = dataclasses.replace(_opening(), on=EX)
    assert apply_splits((entry,), (), (_split(Fraction(2)),)) == ((entry,), ())


def test_two_splits_accumulate() -> None:
    splits = (_split(Fraction(2)), _split(Fraction(3), ex_date=date(2026, 10, 8)))
    (entry,), (out,) = apply_splits((_opening(),), (_fill(),), splits)
    assert (entry.shares, entry.price) == (60, pytest.approx(100 / 6))
    assert (out.filled_shares, out.avg_price) == (60, pytest.approx(100 / 6))


def test_a_fill_between_two_splits_gets_only_the_later_one() -> None:
    splits = (_split(Fraction(2)), _split(Fraction(3), ex_date=date(2026, 10, 8)))
    _, (out,) = apply_splits((), (_fill(at=AFTER),), splits)
    assert (out.filled_shares, out.avg_price) == (30, pytest.approx(100 / 3))


def test_other_tickers_are_untouched() -> None:
    entry, fill = _opening("BBB"), _fill(ticker="BBB")
    assert apply_splits((entry,), (fill,), (_split(Fraction(2)),)) == ((entry,), (fill,))


def test_a_reverse_split_is_exact() -> None:
    _, (out,) = apply_splits((), (_fill(shares=30, price=10.0),), (_split(Fraction(1, 10)),))
    assert (out.filled_shares, out.requested_shares, out.avg_price) == (3, 3, 100.0)


def test_a_non_integral_reverse_split_raises_naming_the_record() -> None:
    with pytest.raises(ValueError, match=r"AAA.*f1"):
        apply_splits((), (_fill(shares=15),), (_split(Fraction(1, 10)),))
    with pytest.raises(ValueError, match=r"opening AAA"):
        apply_splits((_opening(shares=15),), (), (_split(Fraction(1, 10)),))


def test_requested_shares_are_informational_and_floor_instead_of_raising() -> None:
    fill = dataclasses.replace(_fill(shares=10), requested_shares=13)
    _, (out,) = apply_splits((), (fill,), (_split(Fraction(1, 5)),))
    assert (out.requested_shares, out.filled_shares) == (2, 2)


def test_split_window_is_the_sorted_tickers_and_the_earliest_record_date() -> None:
    late_evening = datetime(2026, 9, 1, 2, 0, tzinfo=UTC)  # 22:00 on 08-31 in New York
    opening = (_opening("CCC"),)
    fills = (_fill(ticker="BBB"), _fill("f2", ticker="AAA", at=late_evening))
    assert split_window(opening, fills) == (["AAA", "BBB", "CCC"], date(2026, 8, 31))


def test_split_window_ignores_fills_without_a_date_and_is_none_without_records() -> None:
    undated = dataclasses.replace(_fill(), filled_at=None, avg_price=None)
    assert split_window((), (undated,)) is None


def test_a_zero_share_fill_is_unchanged() -> None:
    fill = dataclasses.replace(
        _fill(), filled_shares=0, avg_price=None, filled_at=None, status=FillStatus.CANCELED
    )
    assert apply_splits((), (fill,), (_split(Fraction(2)),)) == ((), (fill,))


@pytest.mark.parametrize("ratio", [Fraction(0), Fraction(-2)])
def test_a_non_positive_ratio_raises(ratio: Fraction) -> None:
    with pytest.raises(ValueError, match="ratio"):
        apply_splits((), (), (_split(ratio),))


def test_the_new_york_date_decides_which_side_of_the_ex_date() -> None:
    # 23:30 EDT on 10-04 is 03:30 UTC on 10-05: still pre-split in New York.
    late_evening = datetime(2026, 10, 5, 3, 30, tzinfo=UTC)
    _, (out,) = apply_splits((), (_fill(at=late_evening),), (_split(Fraction(2)),))
    assert out.filled_shares == 20


def test_outputs_keep_input_order() -> None:
    fills = (_fill("b", at=AFTER), _fill("a"))
    _, out = apply_splits((), fills, (_split(Fraction(2)),))
    assert [f.client_order_id for f in out] == ["b", "a"]
