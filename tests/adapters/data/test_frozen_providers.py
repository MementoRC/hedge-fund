"""The frozen providers: point-in-time closes, splits and sectors; every raw fact row."""

from __future__ import annotations

from datetime import date
from fractions import Fraction

import pandas as pd
import pytest

from hedge_fund.adapters.data.frozen_providers import (
    FrozenDataProvider,
    FrozenFundamentalsProvider,
)
from hedge_fund.domain.ledger import Split
from hedge_fund.ports.data import DataProvider, DataProviderError
from hedge_fund.ports.fundamentals import FACT_COLUMNS, FundamentalsProvider, coerce_facts
from hedge_fund.services.fundamentals import build_fundamentals_snapshot
from hedge_fund.services.snapshots import build_snapshot
from tests.factories import make_facts, make_frozen_data

DATA = make_frozen_data()  # AAA 2:1 ex 2024-01-22, 0.5 dividend ex 2024-02-01
FIRST = date(2024, 1, 1)
LAST = date(2024, 2, 9)


def _provider(as_of: date) -> FrozenDataProvider:
    return FrozenDataProvider(DATA, as_of)


def test_satisfies_the_port_with_nan_columns_for_unknown_tickers() -> None:
    provider: DataProvider = _provider(LAST)
    closes = provider.closes(["AAA", "ZZZ"], FIRST, LAST)
    assert list(closes.columns) == ["AAA", "ZZZ"]
    assert closes["ZZZ"].isna().all()
    assert closes.index.equals(DATA.closes.index)


def test_closes_keep_the_request_order() -> None:
    closes = _provider(LAST).closes(["SPY", "AAA"], FIRST, LAST)
    assert list(closes.columns) == ["SPY", "AAA"]


def test_closes_before_a_split_are_on_the_pre_split_basis() -> None:
    as_of = date(2024, 1, 19)
    closes = _provider(as_of).closes(["AAA"], FIRST, as_of)["AAA"]
    expected = DATA.closes["AAA"].loc[: pd.Timestamp(as_of)] * 2.0
    pd.testing.assert_series_equal(closes, expected, check_freq=False)


def test_closes_after_the_dividend_are_back_adjusted() -> None:
    closes = _provider(LAST).closes(["AAA"], FIRST, LAST)["AAA"]
    raw = DATA.closes["AAA"]
    factor = 1.0 - 0.5 / raw.loc[pd.Timestamp("2024-01-31")]
    expected = raw.where(raw.index >= pd.Timestamp("2024-02-01"), raw * factor)
    pd.testing.assert_series_equal(closes, expected, check_freq=False)


def test_closes_are_sliced_to_the_window() -> None:
    closes = _provider(LAST).closes(["BBB"], date(2024, 1, 10), date(2024, 1, 12))
    assert [stamp.date() for stamp in closes.index] == [
        date(2024, 1, 10),
        date(2024, 1, 11),
        date(2024, 1, 12),
    ]


def test_closes_ending_after_as_of_fail_closed() -> None:
    with pytest.raises(DataProviderError, match="after as_of"):
        _provider(date(2024, 1, 19)).closes(["AAA"], FIRST, date(2024, 1, 22))


def test_with_as_of_moves_the_view_date_over_the_same_store() -> None:
    later = _provider(date(2024, 1, 19)).with_as_of(date(2024, 1, 22))
    assert later.as_of == date(2024, 1, 22)
    closes = later.closes(["AAA"], date(2024, 1, 19), date(2024, 1, 22))["AAA"]
    stored = DATA.closes["AAA"].loc[pd.Timestamp("2024-01-19") : pd.Timestamp("2024-01-22")]
    assert closes.tolist() == stored.tolist()


def test_an_as_of_outside_the_coverage_is_refused() -> None:
    with pytest.raises(ValueError, match="outside the store"):
        FrozenDataProvider(DATA, date(2024, 2, 12))


def test_splits_are_start_exclusive_and_end_inclusive() -> None:
    provider = _provider(LAST)
    assert provider.splits(["AAA"], date(2024, 1, 21), date(2024, 1, 22)) == DATA.splits
    assert provider.splits(["AAA"], date(2024, 1, 22), LAST) == ()
    assert provider.splits(["BBB"], FIRST, LAST) == ()


def test_splits_ending_after_as_of_fail_closed() -> None:
    with pytest.raises(DataProviderError, match="after as_of"):
        _provider(date(2024, 1, 19)).splits(["AAA"], FIRST, date(2024, 1, 22))


def test_splits_are_ordered_by_ex_date_then_request_order() -> None:
    data = make_frozen_data(
        splits=(
            Split("AAA", date(2024, 1, 22), Fraction(2)),
            Split("BBB", date(2024, 1, 22), Fraction(3)),
            Split("BBB", date(2024, 1, 10), Fraction(1, 2)),
        )
    )
    found = FrozenDataProvider(data, LAST).splits(["BBB", "AAA"], FIRST, LAST)
    assert [(split.ticker, split.ex_date) for split in found] == [
        ("BBB", date(2024, 1, 10)),
        ("BBB", date(2024, 1, 22)),
        ("AAA", date(2024, 1, 22)),
    ]


def test_sectors_come_from_the_store_and_unknowns_are_none() -> None:
    assert _provider(LAST).sectors(["AAA", "CCC"]) == {"AAA": "Energy", "CCC": None}


def test_build_snapshot_over_the_provider_ends_at_as_of_on_its_basis() -> None:
    as_of = date(2024, 1, 19)
    build = build_snapshot(_provider(as_of), ["AAA", "BBB", "CCC"], as_of, lookback=5)
    assert build.snapshot.closes.index[-1].date() == as_of
    assert build.unresolved == {"CCC": "no_sector"}
    last_raw = DATA.closes["AAA"].loc[pd.Timestamp(as_of)]
    assert build.snapshot.closes["AAA"].iloc[-1] == last_raw * 2.0


RESTATED = ("AAA", "us-gaap:Assets", None, "2026-06-30", 1200.0, "2026-09-15", "10-Q/A", "AAA-a")


def test_fundamentals_provider_satisfies_the_port_and_keeps_every_row() -> None:
    provider: FundamentalsProvider = FrozenFundamentalsProvider(DATA)
    facts = provider.facts(["BBB"])
    assert tuple(facts.columns) == FACT_COLUMNS
    assert set(facts["ticker"]) == {"BBB"}
    assert len(facts) == int((DATA.facts["ticker"] == "BBB").sum())


def test_fundamentals_for_unknown_tickers_are_empty() -> None:
    assert FrozenFundamentalsProvider(DATA).facts(["ZZZ"]).empty


def test_a_restatement_is_invisible_before_its_filed_date() -> None:
    original = make_facts(["AAA"])
    restated = pd.DataFrame([RESTATED], columns=list(FACT_COLUMNS))
    facts = coerce_facts(pd.concat([original, restated], ignore_index=True))
    provider = FrozenFundamentalsProvider(make_frozen_data(facts=facts))
    ciks = {"AAA": "0000000001"}

    def assets(as_of: date) -> float:
        build = build_fundamentals_snapshot(provider, ["AAA"], as_of, ciks=ciks)
        return float(build.snapshot.current.loc["AAA", "total_assets"])

    assert assets(date(2026, 9, 15)) == 1100.0  # filed == as_of is not yet visible
    assert assets(date(2026, 9, 16)) == 1200.0
    assert len(provider.facts(["AAA"])) == len(original) + 1
