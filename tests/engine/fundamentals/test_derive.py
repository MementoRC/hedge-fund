"""Tests for point-in-time fundamentals derivation (pure, hand-built facts)."""

from datetime import date
from typing import Any

import pandas as pd
import pytest

from hedge_fund.engine.fundamentals.derive import (
    ANNUAL,
    HALF_YEAR,
    NINE_MONTH,
    QUARTER,
    duration_classes,
    last_filed,
    pit_filter,
    prior_reference,
    reference_period,
    select_versions,
)
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


@pytest.mark.parametrize(
    ("days", "expected"),
    [
        (79, ""),
        (80, QUARTER),
        (100, QUARTER),
        (101, ""),
        (169, ""),
        (170, HALF_YEAR),
        (190, HALF_YEAR),
        (191, ""),
        (259, ""),
        (260, NINE_MONTH),
        (280, NINE_MONTH),
        (281, ""),
        (349, ""),
        (350, ANNUAL),
        (371, ANNUAL),
        (380, ANNUAL),
        (381, ""),
    ],
)
def test_duration_classes_by_inclusive_length(days: int, expected: str) -> None:
    start = pd.Timestamp("2024-01-01")
    end = start + pd.Timedelta(days=days - 1)
    facts = _frame(_fact("Revenues", end.date().isoformat(), 1.0, start="2024-01-01"))
    assert duration_classes(facts).tolist() == [expected]


def test_instant_facts_have_no_duration_class() -> None:
    assert duration_classes(_frame(_fact("Assets", "2024-03-31", 1.0))).tolist() == [""]


def test_reference_period_is_latest_curated_non_dei_end() -> None:
    facts = _frame(
        _fact("Assets", "2024-03-31", 1050.0),
        _fact("Revenues", "2024-03-31", 110.0, start="2024-01-01"),
        _fact("Assets", "2023-12-31", 1000.0, filed="2024-02-15", form="10-K"),
        _fact(SHARES, "2024-04-25", 49.0),
    )
    assert reference_period(facts) == pd.Timestamp("2024-03-31")


def test_reference_period_ignores_filings_not_yet_public() -> None:
    facts = _frame(
        _fact("Assets", "2024-03-31", 1050.0, filed="2024-05-01"),
        _fact("Assets", "2024-06-30", 1100.0, filed="2024-08-01"),
    )
    visible = select_versions(pit_filter(facts, date(2024, 6, 1)))
    assert reference_period(visible) == pd.Timestamp("2024-03-31")


def test_reference_period_is_none_without_curated_non_dei_facts() -> None:
    assert reference_period(_frame(_fact(SHARES, "2024-04-25", 49.0))) is None
    assert reference_period(_frame()) is None


def test_prior_reference_picks_the_nearest_end_within_seven_days() -> None:
    facts = _frame(
        _fact("Assets", "2023-03-25", 1.0, filed="2023-05-01"),
        _fact("Assets", "2023-04-02", 2.0, filed="2023-05-01"),
        _fact("Assets", "2024-03-30", 3.0),
    )
    assert prior_reference(facts, pd.Timestamp("2024-03-30")) == pd.Timestamp("2023-04-02")


def test_prior_reference_tie_prefers_the_earlier_end() -> None:
    facts = _frame(
        _fact("Assets", "2023-04-02", 1.0, filed="2023-05-01"),
        _fact("Assets", "2023-03-27", 2.0, filed="2023-05-01"),
    )
    assert prior_reference(facts, pd.Timestamp("2024-03-30")) == pd.Timestamp("2023-03-27")


def test_prior_reference_is_nat_beyond_seven_days_or_for_dei_only() -> None:
    beyond = _frame(_fact("Assets", "2023-03-22", 1.0, filed="2023-05-01"))
    dei_only = _frame(_fact(SHARES, "2023-03-30", 50.0, filed="2023-05-01"))
    assert pd.isna(prior_reference(beyond, pd.Timestamp("2024-03-30")))
    assert pd.isna(prior_reference(dei_only, pd.Timestamp("2024-03-30")))


def test_prior_reference_is_nat_on_an_empty_frame() -> None:
    assert pd.isna(prior_reference(_frame(), pd.Timestamp("2024-03-30")))


@pytest.mark.parametrize(
    ("end", "prior_end"),
    [
        ("2024-02-29", "2023-02-28"),  # leap day: E - 1 year clips to Feb 28
        ("2025-03-01", "2024-03-01"),  # the year back contains a leap day
    ],
)
def test_prior_reference_across_leap_years(end: str, prior_end: str) -> None:
    facts = _frame(
        _fact("Assets", prior_end, 1.0, filed="2023-05-01"),
        _fact("Assets", end, 2.0),
    )
    assert prior_reference(facts, pd.Timestamp(end)) == pd.Timestamp(prior_end)


def test_last_filed_is_the_latest_filing_among_visible_curated_facts() -> None:
    facts = _frame(
        _fact("Assets", "2023-12-31", 1000.0, filed="2024-02-15", form="10-K"),
        _fact("Assets", "2024-03-31", 1050.0, filed="2024-05-01"),
        _fact(SHARES, "2024-05-05", 49.0, filed="2024-05-10", form="10-K/A"),
        # Uncurated tag: pit_filter drops it, so its later filing date does not count.
        _fact("NotACuratedTag", "2024-03-31", 5.0, filed="2024-05-20"),
        _fact("Assets", "2024-06-30", 1100.0, filed="2024-06-05"),
    )
    assert last_filed(pit_filter(facts, date(2024, 6, 1))) == pd.Timestamp("2024-05-10")


def test_last_filed_is_nat_on_an_empty_frame() -> None:
    assert pd.isna(last_filed(_frame()))
