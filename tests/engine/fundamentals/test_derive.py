"""Tests for point-in-time fundamentals derivation (pure, hand-built facts)."""

from datetime import date
from typing import Any

import pandas as pd

from hedge_fund.engine.fundamentals.derive import pit_filter, select_versions
from hedge_fund.ports.fundamentals import FACT_COLUMNS, coerce_facts, empty_facts

SHARES = "dei:EntityCommonStockSharesOutstanding"


def _fact(
    tag: str,
    end: str,
    value: float,
    *,
    start: str | None = None,
    filed: str = "2024-05-01",
    form: str = "10-Q",
    accn: str | None = None,
    ticker: str = "AAA",
) -> dict[str, Any]:
    """One fact row; bare tags get the us-gaap prefix and accn defaults to ticker + filed."""
    return {
        "ticker": ticker,
        "tag": tag if ":" in tag else f"us-gaap:{tag}",
        "start": start,
        "end": end,
        "value": value,
        "filed": filed,
        "form": form,
        "accn": accn if accn is not None else f"{ticker}-{filed}",
    }


def _frame(*rows: dict[str, Any]) -> pd.DataFrame:
    return coerce_facts(pd.DataFrame(list(rows), columns=list(FACT_COLUMNS)))


def test_pit_filter_is_strict_on_the_filing_date() -> None:
    facts = _frame(_fact("Assets", "2024-03-31", 1.0, filed="2024-05-01"))
    assert pit_filter(facts, date(2024, 5, 1)).empty
    assert len(pit_filter(facts, date(2024, 5, 2))) == 1


def test_pit_filter_drops_later_filings_uncurated_tags_and_other_forms() -> None:
    facts = _frame(
        _fact("Assets", "2024-03-31", 1.0, filed="2024-05-01"),
        _fact("Assets", "2024-06-30", 2.0, filed="2024-08-01"),
        _fact("Goodwill", "2024-03-31", 3.0, filed="2024-05-01"),
        _fact("Assets", "2024-03-31", 4.0, filed="2024-05-01", form="8-K"),
    )
    assert pit_filter(facts, date(2024, 6, 1))["value"].tolist() == [1.0]


def test_restatement_filed_on_or_after_as_of_is_ignored_and_before_is_used() -> None:
    facts = _frame(
        _fact("Revenues", "2023-03-31", 100.0, start="2023-01-01", filed="2023-05-01"),
        _fact("Revenues", "2023-03-31", 105.0, start="2023-01-01", filed="2024-05-01"),
    )
    on_as_of = select_versions(pit_filter(facts, date(2024, 5, 1)))
    after = select_versions(pit_filter(facts, date(2024, 5, 2)))
    assert on_as_of["value"].tolist() == [100.0]
    assert after["value"].tolist() == [105.0]


def test_version_selection_keeps_one_fact_per_tag_start_end() -> None:
    facts = _frame(
        _fact("Revenues", "2023-03-31", 100.0, start="2023-01-01", filed="2023-05-01"),
        _fact("Revenues", "2023-03-31", 105.0, start="2023-01-01", filed="2024-05-01"),
        _fact("Revenues", "2023-06-30", 200.0, start="2023-01-01", filed="2023-08-01"),
        _fact("Revenues", "2023-06-30", 100.0, start="2023-04-01", filed="2023-08-01"),
    )
    assert sorted(select_versions(facts)["value"].tolist()) == [100.0, 105.0, 200.0]


def test_instant_facts_survive_version_selection() -> None:
    facts = _frame(
        _fact("Assets", "2023-12-31", 1000.0, filed="2024-02-15", form="10-K"),
        _fact("Assets", "2023-12-31", 1010.0, filed="2025-02-15", form="10-K"),
        _fact(SHARES, "2024-02-01", 50.0, filed="2024-02-15", form="10-K"),
    )
    versioned = select_versions(facts)
    assert versioned["start"].isna().all()
    assert sorted(versioned["value"].tolist()) == [50.0, 1010.0]


def test_version_selection_keeps_each_tickers_own_latest_version() -> None:
    facts = _frame(
        _fact("Revenues", "2023-03-31", 1.0, start="2023-01-01", filed="2023-05-01", ticker="AAA"),
        _fact("Revenues", "2023-03-31", 2.0, start="2023-01-01", filed="2024-05-01", ticker="AAA"),
        _fact("Revenues", "2023-03-31", 10.0, start="2023-01-01", filed="2023-05-02", ticker="BBB"),
        _fact("Revenues", "2023-03-31", 20.0, start="2023-01-01", filed="2023-06-01", ticker="BBB"),
    )
    selected = select_versions(facts)
    assert selected["ticker"].tolist() == ["AAA", "BBB"]
    assert selected["value"].tolist() == [2.0, 20.0]


def test_empty_facts_give_empty_frames_with_canonical_columns() -> None:
    empty = empty_facts()
    for result in (pit_filter(empty, date(2024, 5, 1)), select_versions(empty)):
        assert result.empty
        assert list(result.columns) == list(FACT_COLUMNS)


def test_version_selection_is_sorted_by_ticker_tag_end_start_with_a_fresh_index() -> None:
    facts = _frame(
        _fact("Revenues", "2023-06-30", 1.0, start="2023-04-01", ticker="BBB"),
        _fact("Revenues", "2023-06-30", 2.0, start="2023-01-01", ticker="AAA"),
        _fact("Revenues", "2023-03-31", 3.0, start="2023-01-01", ticker="AAA"),
        _fact("Assets", "2023-06-30", 4.0, ticker="AAA"),
        _fact("Revenues", "2023-06-30", 5.0, start="2023-04-01", ticker="AAA"),
    )
    selected = select_versions(facts)
    assert selected["value"].tolist() == [4.0, 3.0, 2.0, 5.0, 1.0]
    assert selected.index.tolist() == list(range(5))


def test_same_day_versions_break_ties_by_accession() -> None:
    facts = _frame(
        _fact("Assets", "2023-12-31", 1.0, filed="2024-02-15", accn="0001-24-000002"),
        _fact("Assets", "2023-12-31", 2.0, filed="2024-02-15", accn="0001-24-000001"),
    )
    assert select_versions(facts)["value"].tolist() == [1.0]
