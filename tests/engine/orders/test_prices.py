"""order_prices: last finite close per ticker, no older than max_age_days, or one ValueError."""

from __future__ import annotations

import math
import re
from datetime import date

import pandas as pd
import pytest

from hedge_fund.engine.orders.prices import order_prices

AS_OF = date(2026, 9, 30)  # Wednesday
NAN = math.nan
INDEX = pd.to_datetime(["2026-09-21", "2026-09-22", "2026-09-23", "2026-09-29", "2026-09-30"])
CLOSES = pd.DataFrame(
    {
        "AAA": [10.0, 11.0, 12.0, 13.0, 14.0],  # last close on as_of
        "BBB": [20.0, 21.0, 22.0, NAN, NAN],  # NaN tail: last close 2026-09-23, 7 days old
        "CCC": [30.0, 31.0, NAN, NAN, NAN],  # last close 2026-09-22, 8 days old: stale
        "DDD": [NAN] * 5,  # no finite close
        "EEE": [40.0, 41.0, 42.0, math.inf, NAN],  # inf is not a close
    },
    index=INDEX,
)


def test_last_finite_close_and_its_date() -> None:
    assert order_prices(CLOSES, ["AAA", "BBB", "EEE"], as_of=AS_OF) == {
        "AAA": (14.0, date(2026, 9, 30)),
        "BBB": (22.0, date(2026, 9, 23)),
        "EEE": (42.0, date(2026, 9, 23)),
    }


def test_values_are_floats_and_dates_are_dates() -> None:
    [(price, day)] = order_prices(CLOSES, ["AAA"], as_of=AS_OF).values()
    assert type(price) is float
    assert type(day) is date


def test_no_tickers_gives_no_prices() -> None:
    assert order_prices(CLOSES, [], as_of=AS_OF) == {}


def test_unsorted_rows_still_give_the_latest_close() -> None:
    shuffled = CLOSES.iloc[[4, 0, 3, 1, 2]]
    assert order_prices(shuffled, ["AAA"], as_of=AS_OF) == {"AAA": (14.0, date(2026, 9, 30))}


@pytest.mark.parametrize(
    ("ticker", "message"),
    [
        pytest.param("CCC", "CCC (last close 2026-09-22)", id="stale"),
        pytest.param("DDD", "DDD (no finite close)", id="all-nan"),
        pytest.param("ZZZ", "ZZZ (no finite close)", id="absent"),
    ],
)
def test_a_ticker_without_a_usable_price_raises(ticker: str, message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        order_prices(CLOSES, ["AAA", ticker], as_of=AS_OF)


def test_one_error_names_every_bad_ticker_sorted() -> None:
    with pytest.raises(ValueError, match="no usable order price") as info:
        order_prices(CLOSES, ["ZZZ", "AAA", "DDD", "CCC"], as_of=AS_OF)
    assert str(info.value) == (
        "no usable order price (missing, or last close before 2026-09-23) for: "
        "CCC (last close 2026-09-22), DDD (no finite close), ZZZ (no finite close)"
    )


def test_an_empty_column_has_no_price() -> None:
    empty = pd.DataFrame({"AAA": pd.Series([], dtype="float64")}, index=pd.DatetimeIndex([]))
    with pytest.raises(ValueError, match=re.escape("AAA (no finite close)")):
        order_prices(empty, ["AAA"], as_of=AS_OF)


@pytest.mark.parametrize(
    ("max_age_days", "accepted"),
    [
        pytest.param(8, True, id="8-days-accepts-an-8-day-old-close"),
        pytest.param(7, False, id="default-limit"),
        pytest.param(0, False, id="0-days"),
    ],
)
def test_max_age_days_overrides_the_default(max_age_days: int, accepted: bool) -> None:
    if accepted:
        prices = order_prices(CLOSES, ["CCC"], as_of=AS_OF, max_age_days=max_age_days)
        assert prices == {"CCC": (31.0, date(2026, 9, 22))}
    else:
        with pytest.raises(ValueError, match="CCC"):
            order_prices(CLOSES, ["CCC"], as_of=AS_OF, max_age_days=max_age_days)


def test_zero_max_age_accepts_a_close_on_as_of() -> None:
    assert order_prices(CLOSES, ["AAA"], as_of=AS_OF, max_age_days=0) == {
        "AAA": (14.0, date(2026, 9, 30))
    }


def test_the_input_frame_is_not_mutated() -> None:
    before = CLOSES.copy()
    order_prices(CLOSES, ["AAA", "BBB", "EEE"], as_of=AS_OF)
    pd.testing.assert_frame_equal(CLOSES, before)


def test_package_exports() -> None:
    import hedge_fund.engine.orders as orders

    assert "order_prices" in orders.__all__
    assert orders.order_prices is order_prices
