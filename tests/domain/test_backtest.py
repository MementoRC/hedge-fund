"""Tests for the backtest domain types."""

from __future__ import annotations

import dataclasses
import math
from datetime import date

import pandas as pd
import pytest

from hedge_fund.domain.backtest import (
    RESIDUAL_BIASES,
    MemberSpan,
    SimCosts,
    SimEvent,
    SimEventKind,
)
from tests.factories import make_frozen_data


def test_residual_biases_name_each_known_limitation() -> None:
    assert RESIDUAL_BIASES
    assert isinstance(RESIDUAL_BIASES, tuple)
    assert all(isinstance(entry, str) for entry in RESIDUAL_BIASES)
    for keyword in (
        "Sectors",
        "yfinance",
        "Membership",
        "Late removals",
        "Ticker reuse",
        "breakers",
    ):
        assert any(keyword in entry for entry in RESIDUAL_BIASES), keyword


def test_sim_costs_defaults() -> None:
    costs = SimCosts()
    assert costs.slippage_bps == 5.0
    assert costs.borrow_rate == 0.005


@pytest.mark.parametrize("field", ["slippage_bps", "borrow_rate"])
@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf])
def test_sim_costs_reject_negative_or_non_finite(field: str, value: float) -> None:
    with pytest.raises(ValueError, match="must be finite and >= 0"):
        SimCosts(**{field: value})


def test_sim_event_defaults_and_kind_values() -> None:
    event = SimEvent(date(2024, 1, 2), SimEventKind.FILL, "AAA", 10, 100.0, -1000.0)
    assert event.cost == 0.0
    assert event.client_order_id is None
    assert event.reason is None
    assert [kind.value for kind in SimEventKind] == [
        "fill",
        "reject",
        "split",
        "dividend",
        "delist",
        "borrow",
    ]


def test_a_span_contains_its_start_and_not_its_end() -> None:
    span = MemberSpan("AAA", date(2024, 1, 2), date(2024, 3, 15))
    assert span.contains(date(2024, 1, 2))
    assert span.contains(date(2024, 3, 14))
    assert not span.contains(date(2024, 3, 15))
    assert not span.contains(date(2024, 1, 1))


def test_an_open_span_contains_every_later_day() -> None:
    assert MemberSpan("AAA", date(2024, 1, 2), None).contains(date(2099, 1, 1))


def test_overlaps_is_inclusive_of_the_window() -> None:
    span = MemberSpan("AAA", date(2024, 1, 2), date(2024, 3, 15))
    assert span.overlaps(date(2024, 3, 14), date(2024, 12, 31))
    assert not span.overlaps(date(2024, 3, 15), date(2024, 12, 31))
    assert span.overlaps(date(2023, 1, 1), date(2024, 1, 2))
    assert not span.overlaps(date(2023, 1, 1), date(2024, 1, 1))


def test_an_empty_span_is_rejected() -> None:
    with pytest.raises(ValueError, match="not after start"):
        MemberSpan("AAA", date(2024, 3, 1), date(2024, 3, 1))


def test_frozen_data_rejects_a_tz_aware_index() -> None:
    data = make_frozen_data()
    aware = data.closes.tz_localize("UTC")
    with pytest.raises(ValueError, match="tz-naive"):
        dataclasses.replace(data, closes=aware, opens=aware)


def test_frozen_data_rejects_unsorted_dates() -> None:
    data = make_frozen_data()
    flipped = data.closes.iloc[::-1]
    with pytest.raises(ValueError, match="sorted and unique"):
        dataclasses.replace(data, closes=flipped, opens=flipped)


def test_frozen_data_rejects_opens_that_do_not_match_the_closes() -> None:
    data = make_frozen_data()
    with pytest.raises(ValueError, match="share dates and tickers"):
        dataclasses.replace(data, opens=data.opens.drop(columns=["SPY"]))


def test_frozen_data_rejects_prices_outside_its_coverage() -> None:
    with pytest.raises(ValueError, match="outside"):
        dataclasses.replace(make_frozen_data(), end=date(2024, 2, 8))


def _macro_cases() -> list[tuple[str, object, str]]:
    macro = make_frozen_data().macro
    return [
        (
            "not-a-datetime-index",
            macro.reset_index(drop=True),
            "macro index must be a DatetimeIndex",
        ),
        ("unsorted", macro.iloc[::-1], "macro index must be sorted and unique"),
        ("duplicate", pd.concat([macro, macro.iloc[:1]]).sort_index(), "sorted and unique"),
        (
            "after-end",
            macro.set_axis(macro.index + pd.Timedelta(days=30)),
            "macro dates fall outside",
        ),
        (
            "before-start",
            macro.set_axis(macro.index - pd.Timedelta(days=30)),
            "macro dates fall outside",
        ),
        ("no-dgs10", macro.drop(columns=["DGS10"]), r"macro lacks series \['DGS10'\]"),
        ("no-vixcls", macro.drop(columns=["VIXCLS"]), r"macro lacks series \['VIXCLS'\]"),
        ("integer", macro.astype({"DGS10": "int64"}), "macro values must be float"),
        ("tz-aware", macro.tz_localize("UTC"), "macro index must be tz-naive"),
        ("infinite", macro.assign(DGS10=float("inf")), "macro values must be finite"),
        ("negative-infinite", macro.assign(VIXCLS=float("-inf")), "macro values must be finite"),
    ]


@pytest.mark.parametrize(
    ("bad", "message"),
    [case[1:] for case in _macro_cases()],
    ids=[case[0] for case in _macro_cases()],
)
def test_frozen_data_rejects_bad_macro(bad: pd.DataFrame, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(make_frozen_data(), macro=bad)


def test_frozen_data_accepts_macro_dated_exactly_on_start_and_end_and_nan_gaps() -> None:
    data = make_frozen_data()
    edges = data.macro.iloc[[0, -1]].copy()
    edges.iloc[0, 0] = float("nan")
    assert edges.index[0].date() == data.start
    assert edges.index[-1].date() == data.end
    assert dataclasses.replace(data, macro=edges).macro is edges


def test_frozen_data_accepts_macro_with_extra_series_and_its_own_dates() -> None:
    data = make_frozen_data()
    extra = data.macro.iloc[::2].assign(DFF=5.3)
    assert dataclasses.replace(data, macro=extra).macro is extra
