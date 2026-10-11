"""Tests for point-in-time fundamentals derivation (pure, hand-built facts)."""

import math
from datetime import date, timedelta
from typing import Any

import pandas as pd
import pytest

from hedge_fund.domain.fundamentals import METRIC_NAMES, PERIOD_COLUMNS
from hedge_fund.engine.fundamentals.derive import (
    ANNUAL,
    HALF_YEAR,
    MAX_STALENESS_DAYS,
    NINE_MONTH,
    NO_FILINGS,
    NO_REFERENCE_PERIOD,
    QUARTER,
    STALE_FUNDAMENTALS,
    current_shares,
    derive_fundamentals,
    duration_classes,
    first_complete,
    flow_ttm,
    last_filed,
    pit_filter,
    prior_reference,
    prior_shares,
    reference_period,
    select_versions,
    stock_value,
)
from hedge_fund.engine.fundamentals.metrics import METRICS, MetricSpec
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


def _spec(name: str) -> MetricSpec:
    return next(spec for spec in METRICS if spec.name == name)


def _calendar_revenue(tag: str = "Revenues") -> list[dict[str, Any]]:
    """Calendar fiscal years 2022-2024 with YTD 10-Qs, a Q2 discrete quarter and 10-Ks."""
    return [
        _fact(tag, "2022-12-31", 360.0, start="2022-01-01", filed="2023-02-15", form="10-K"),
        _fact(tag, "2023-03-31", 100.0, start="2023-01-01", filed="2023-05-01"),
        _fact(tag, "2023-06-30", 200.0, start="2023-01-01", filed="2023-08-01"),
        _fact(tag, "2023-09-30", 300.0, start="2023-01-01", filed="2023-11-01"),
        _fact(tag, "2023-12-31", 400.0, start="2023-01-01", filed="2024-02-15", form="10-K"),
        _fact(tag, "2024-03-31", 110.0, start="2024-01-01", filed="2024-05-01"),
        _fact(tag, "2024-06-30", 120.0, start="2024-04-01", filed="2024-08-01"),
        _fact(tag, "2024-06-30", 230.0, start="2024-01-01", filed="2024-08-01"),
        _fact(tag, "2024-09-30", 360.0, start="2024-01-01", filed="2024-11-01"),
        _fact(tag, "2024-12-31", 500.0, start="2024-01-01", filed="2025-02-15", form="10-K"),
    ]


@pytest.mark.parametrize(
    ("end", "expected"),
    [
        ("2024-03-31", 410.0),  # Q1: YTD 110 + FY2023 400 - Q1 2023 100
        ("2024-06-30", 430.0),  # Q2: 6M 230 + 400 - 6M 2023 200 (3M discrete 120 unused)
        ("2024-09-30", 460.0),  # Q3: 9M 360 + 400 - 9M 2023 300
        ("2024-12-31", 500.0),  # fiscal year end: the annual value
    ],
)
def test_ttm_identity_at_each_quarter(end: str, expected: float) -> None:
    assert flow_ttm(_frame(*_calendar_revenue()), pd.Timestamp(end)) == pytest.approx(expected)


def test_ttm_is_nan_when_a_piece_is_missing() -> None:
    # At 2023-06-30 the 6M YTD of 2022 was never reported.
    assert math.isnan(flow_ttm(_frame(*_calendar_revenue()), pd.Timestamp("2023-06-30")))


@pytest.mark.parametrize(
    "dropped",
    [7, 4, 2],  # YTD(E) 6M 2024, FY(prior) 2023, YTD(E - 1 year) 6M 2023
)
def test_ttm_is_nan_without_each_piece_of_the_identity(dropped: int) -> None:
    rows = _calendar_revenue()
    del rows[dropped]
    assert math.isnan(flow_ttm(_frame(*rows), pd.Timestamp("2024-06-30")))


def test_ttm_is_nan_on_empty_facts_and_at_nat() -> None:
    assert math.isnan(flow_ttm(_frame(), pd.Timestamp("2024-06-30")))
    assert math.isnan(flow_ttm(_frame(*_calendar_revenue()), pd.NaT))
    assert math.isnan(first_complete(_frame(), _spec("revenue").tags, pd.NaT, flow_ttm))


def test_annual_fact_at_e_takes_precedence_over_any_ytd() -> None:
    facts = _frame(
        _fact("Revenues", "2024-12-31", 500.0, start="2024-01-01", form="10-K"),
        _fact("Revenues", "2024-12-31", 999.0, start="2024-04-01"),
    )
    assert flow_ttm(facts, pd.Timestamp("2024-12-31")) == 500.0


def test_annual_fact_within_seven_days_beats_a_complete_ytd_identity() -> None:
    facts = _frame(
        _fact("Revenues", "2024-12-27", 500.0, start="2023-12-28", form="10-K"),  # E - 4 days
        # A complete nine-month identity at E would give 999 + 800 - 700 = 1099.
        _fact("Revenues", "2024-12-31", 999.0, start="2024-04-01"),
        _fact("Revenues", "2024-03-31", 800.0, start="2023-04-01", form="10-K"),
        _fact("Revenues", "2023-12-31", 700.0, start="2023-04-01"),
    )
    assert flow_ttm(facts, pd.Timestamp("2024-12-31")) == 500.0


def test_discrete_q2_quarter_is_not_used_as_ytd() -> None:
    facts = _frame(
        _fact("Revenues", "2023-12-31", 400.0, start="2023-01-01", form="10-K"),
        _fact("Revenues", "2024-06-30", 120.0, start="2024-04-01"),
        _fact("Revenues", "2023-06-30", 100.0, start="2023-04-01"),
    )
    assert math.isnan(flow_ttm(facts, pd.Timestamp("2024-06-30")))


def test_quarter_counts_as_ytd_only_right_after_an_annual_fact() -> None:
    facts = _frame(
        _fact("Revenues", "2023-12-31", 400.0, start="2023-01-01", form="10-K"),
        _fact("Revenues", "2024-03-28", 110.0, start="2024-01-01"),  # Q1: opens the year
        # Nearer E, but start - 1 day is 9 days after the annual end: not a YTD candidate.
        _fact("Revenues", "2024-03-31", 999.0, start="2024-01-10"),
        _fact("Revenues", "2023-03-31", 100.0, start="2023-01-01"),
    )
    assert flow_ttm(facts, pd.Timestamp("2024-03-31")) == pytest.approx(410.0)


def test_fy_prior_must_be_annual_class() -> None:
    facts = _frame(
        _fact("Revenues", "2024-06-30", 230.0, start="2024-01-01"),
        _fact("Revenues", "2023-06-30", 200.0, start="2023-01-01"),
        _fact("Revenues", "2023-12-31", 400.0, start="2023-04-01", form="10-K"),  # 275 days
    )
    assert math.isnan(flow_ttm(facts, pd.Timestamp("2024-06-30")))


def test_prior_ytd_start_must_match_within_seven_days() -> None:
    facts = _frame(
        _fact("Revenues", "2024-06-30", 230.0, start="2024-01-01"),
        _fact("Revenues", "2023-12-31", 400.0, start="2023-01-01", form="10-K"),
        _fact("Revenues", "2023-06-30", 200.0, start="2023-01-12"),  # 170 days, start +11
    )
    assert math.isnan(flow_ttm(facts, pd.Timestamp("2024-06-30")))


def test_nearest_end_wins_among_matches_within_seven_days() -> None:
    facts = _frame(
        _fact("Revenues", "2024-06-30", 230.0, start="2024-01-01"),
        _fact("Revenues", "2023-12-31", 400.0, start="2023-01-01", form="10-K"),
        _fact("Revenues", "2023-06-25", 190.0, start="2023-01-01"),
        _fact("Revenues", "2023-06-30", 200.0, start="2023-01-01"),
    )
    assert flow_ttm(facts, pd.Timestamp("2024-06-30")) == pytest.approx(430.0)


def test_nearest_end_wins_for_ytd_and_fy_prior() -> None:
    facts = _frame(
        _fact("Revenues", "2024-06-26", 225.0, start="2024-01-01"),  # E - 4 days
        _fact("Revenues", "2024-06-30", 230.0, start="2024-01-01"),
        _fact("Revenues", "2023-12-27", 390.0, start="2023-01-05", form="10-K"),  # 4 days early
        _fact("Revenues", "2023-12-31", 400.0, start="2023-01-01", form="10-K"),
        _fact("Revenues", "2023-06-30", 200.0, start="2023-01-01"),
    )
    assert flow_ttm(facts, pd.Timestamp("2024-06-30")) == pytest.approx(430.0)


@pytest.mark.parametrize("reverse", [False, True])
def test_equal_distance_tie_prefers_the_earlier_end_in_any_row_order(reverse: bool) -> None:
    rows = [
        _fact("Revenues", "2024-12-28", 500.0, start="2024-01-01", form="10-K"),  # E - 3 days
        _fact("Revenues", "2025-01-03", 600.0, start="2024-01-05", form="10-K"),  # E + 3 days
    ]
    facts = _frame(*(reversed(rows) if reverse else rows))
    assert flow_ttm(facts, pd.Timestamp("2024-12-31")) == 500.0


def _fiscal_year_change_revenue() -> list[dict[str, Any]]:
    """December FYE 2023, a 182-day Jan-Jun 2024 transition period (10-KT), then June FYE."""
    return [
        _fact("Revenues", "2023-06-30", 200.0, start="2023-01-01", filed="2023-08-01"),
        _fact("Revenues", "2023-12-31", 400.0, start="2023-01-01", filed="2024-02-15", form="10-K"),
        _fact(
            "Revenues", "2024-06-30", 250.0, start="2024-01-01", filed="2024-08-15", form="10-KT"
        ),
        _fact("Revenues", "2024-09-30", 130.0, start="2024-07-01", filed="2024-11-01"),
        _fact("Revenues", "2024-12-31", 270.0, start="2024-07-01", filed="2025-02-01"),
        _fact("Revenues", "2025-03-31", 400.0, start="2024-07-01", filed="2025-05-01"),
        _fact("Revenues", "2025-06-30", 540.0, start="2024-07-01", filed="2025-08-15", form="10-K"),
    ]


@pytest.mark.parametrize(
    ("end", "expected"),
    [
        ("2024-06-30", 250.0 + 400.0 - 200.0),  # calendar 6M identity still forms
        ("2024-09-30", math.nan),  # new Q1 does not follow an annual fact
        ("2024-12-31", math.nan),  # no annual FY(prior) ends at 2024-06-30
        ("2025-03-31", math.nan),
        ("2025-06-30", 540.0),  # first June fiscal year: the annual value
    ],
)
def test_ttm_across_a_fiscal_year_change(end: str, expected: float) -> None:
    value = flow_ttm(_frame(*_fiscal_year_change_revenue()), pd.Timestamp(end))
    if math.isnan(expected):
        assert math.isnan(value)
    else:
        assert value == pytest.approx(expected)


def test_10k_with_annual_and_discrete_q4_gives_the_annual_value() -> None:
    facts = _frame(
        _fact("Revenues", "2024-09-30", 360.0, start="2024-01-01", filed="2024-11-01"),
        _fact("Revenues", "2024-12-31", 500.0, start="2024-01-01", filed="2025-02-15", form="10-K"),
        _fact("Revenues", "2024-12-31", 140.0, start="2024-10-01", filed="2025-02-15", form="10-K"),
    )
    assert flow_ttm(facts, pd.Timestamp("2024-12-31")) == 500.0


def _fiscal_52_53_week_revenue() -> list[dict[str, Any]]:
    """Apple-style fiscal years: 53-week FY2023 (371 days), 52-week FY2024 (364 days)."""
    return [
        _fact("Revenues", "2023-09-30", 383.0, start="2022-09-25", filed="2023-11-03", form="10-K"),
        _fact("Revenues", "2022-12-31", 117.0, start="2022-09-25", filed="2023-02-03"),  # 98 d
        _fact("Revenues", "2023-04-01", 212.0, start="2022-09-25", filed="2023-05-05"),  # 189 d
        _fact("Revenues", "2023-07-01", 294.0, start="2022-09-25", filed="2023-08-04"),  # 280 d
        _fact("Revenues", "2023-12-30", 120.0, start="2023-10-01", filed="2024-02-02"),  # 91 d
        _fact("Revenues", "2024-03-30", 211.0, start="2023-10-01", filed="2024-05-03"),  # 182 d
        _fact("Revenues", "2024-06-29", 297.0, start="2023-10-01", filed="2024-08-02"),  # 273 d
        _fact("Revenues", "2024-09-28", 391.0, start="2023-10-01", filed="2024-11-01", form="10-K"),
    ]


@pytest.mark.parametrize(
    ("end", "expected"),
    [
        ("2023-12-30", 120.0 + 383.0 - 117.0),  # prior Q1 ends +1 day, starts -6 days
        ("2024-03-30", 211.0 + 383.0 - 212.0),  # prior 6M ends +2 days
        ("2024-06-29", 297.0 + 383.0 - 294.0),  # prior 9M ends +2 days
        ("2024-09-28", 391.0),  # 52-week fiscal year end
    ],
)
def test_ttm_with_52_53_week_fiscal_years(end: str, expected: float) -> None:
    facts = _frame(*_fiscal_52_53_week_revenue())
    assert flow_ttm(facts, pd.Timestamp(end)) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("end", "expected"),
    [
        ("2024-03-31", 370.0),  # E' = 2023-03-31: Q1 100 + FY2022 360 - Q1 2022 90
        ("2024-12-31", 400.0),  # E' = 2023-12-31: the FY2023 annual value
    ],
)
def test_prior_year_ttm_is_the_same_computation_at_e_prime(end: str, expected: float) -> None:
    facts = _frame(
        *_calendar_revenue(),
        _fact("Revenues", "2022-03-31", 90.0, start="2022-01-01", filed="2022-05-01"),
    )
    prior_end = prior_reference(facts, pd.Timestamp(end))
    assert flow_ttm(facts, prior_end) == pytest.approx(expected)


def test_tag_fallback_across_the_asc_606_switch() -> None:
    facts = _frame(
        _fact("SalesRevenueNet", "2017-12-31", 900.0, start="2017-01-01", form="10-K"),
        _fact(
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "2018-12-31",
            1000.0,
            start="2018-01-01",
            form="10-K",
        ),
    )
    tags = _spec("revenue").tags
    assert first_complete(facts, tags, pd.Timestamp("2018-12-31"), flow_ttm) == 1000.0
    assert first_complete(facts, tags, pd.Timestamp("2017-12-31"), flow_ttm) == 900.0


def test_first_tag_with_a_complete_value_wins() -> None:
    facts = _frame(
        *_calendar_revenue("SalesRevenueNet"),
        _fact("Revenues", "2024-12-31", 777.0, start="2024-01-01", form="10-K"),
    )
    tags = _spec("revenue").tags
    assert first_complete(facts, tags, pd.Timestamp("2024-12-31"), flow_ttm) == 777.0
    # Revenues is incomplete at Q2, so the lower-priority tag supplies every piece.
    value = first_complete(facts, tags, pd.Timestamp("2024-06-30"), flow_ttm)
    assert value == pytest.approx(430.0)


def test_ttm_pieces_never_mix_tags() -> None:
    mixed = _frame(
        _fact("Revenues", "2024-06-30", 230.0, start="2024-01-01"),
        _fact("Revenues", "2023-06-30", 200.0, start="2023-01-01"),
        _fact("SalesRevenueNet", "2023-12-31", 400.0, start="2023-01-01", form="10-K"),
    )
    tags = _spec("revenue").tags
    assert math.isnan(first_complete(mixed, tags, pd.Timestamp("2024-06-30"), flow_ttm))
    complete_later_tag = pd.concat(
        [mixed, _frame(*_calendar_revenue("RevenueFromContractWithCustomerExcludingAssessedTax"))],
        ignore_index=True,
    )
    value = first_complete(complete_later_tag, tags, pd.Timestamp("2024-06-30"), flow_ttm)
    assert value == pytest.approx(430.0)


def test_stock_value_at_instant_within_seven_days() -> None:
    facts = _frame(
        _fact("Assets", "2024-03-30", 1050.0),
        _fact("Assets", "2023-12-31", 1000.0, filed="2024-02-15", form="10-K"),
    )
    assert stock_value(facts, pd.Timestamp("2024-03-31")) == 1050.0
    assert stock_value(facts, pd.Timestamp("2023-12-31")) == 1000.0
    assert math.isnan(stock_value(facts, pd.Timestamp("2024-06-30")))


def test_stock_value_ignores_duration_facts() -> None:
    facts = _frame(_fact("Assets", "2024-03-31", 5.0, start="2024-01-01"))
    assert math.isnan(stock_value(facts, pd.Timestamp("2024-03-31")))


@pytest.mark.parametrize("reverse", [False, True])
def test_stock_value_nearest_instant_wins_and_ties_prefer_the_earlier_end(reverse: bool) -> None:
    rows = [
        _fact("Assets", "2024-03-26", 1.0),  # E - 5 days
        _fact("Assets", "2024-03-28", 2.0),  # E - 3 days: tie with E + 3, earlier wins
        _fact("Assets", "2024-04-03", 3.0),  # E + 3 days
    ]
    facts = _frame(*(reversed(rows) if reverse else rows))
    assert stock_value(facts, pd.Timestamp("2024-03-31")) == 2.0
    assert stock_value(facts, pd.Timestamp("2024-03-25")) == 1.0


def test_stock_fallback_tag_is_used_per_reference_period() -> None:
    facts = _frame(
        _fact("DebtCurrent", "2024-03-31", 70.0),
        _fact("LongTermDebtCurrent", "2023-12-31", 50.0, filed="2024-02-15", form="10-K"),
    )
    tags = _spec("debt_current").tags
    assert first_complete(facts, tags, pd.Timestamp("2024-03-31"), stock_value) == 70.0
    assert first_complete(facts, tags, pd.Timestamp("2023-12-31"), stock_value) == 50.0


def test_first_stock_tag_in_priority_order_with_a_value_wins() -> None:
    facts = _frame(
        _fact("DebtCurrent", "2024-03-31", 70.0),
        _fact("LongTermDebtCurrent", "2024-03-29", 50.0),
    )
    tags = _spec("debt_current").tags
    assert first_complete(facts, tags, pd.Timestamp("2024-03-31"), stock_value) == 50.0


def _share_history() -> pd.DataFrame:
    """Q1 2023 first reported in accn q1-23, then repeated (and versioned) by q1-24."""
    return _frame(
        _fact(
            "Revenues", "2023-03-31", 100.0, start="2023-01-01", filed="2023-05-01", accn="q1-23"
        ),
        _fact(SHARES, "2023-04-25", 52.0, filed="2023-05-01", accn="q1-23"),
        _fact(
            "Revenues", "2023-03-31", 100.0, start="2023-01-01", filed="2024-05-01", accn="q1-24"
        ),
        _fact(
            "Revenues", "2024-03-31", 110.0, start="2024-01-01", filed="2024-05-01", accn="q1-24"
        ),
        _fact(SHARES, "2024-04-25", 49.0, filed="2024-05-01", accn="q1-24"),
    )


def test_current_shares_come_from_the_latest_filing_with_a_cover_count() -> None:
    assert current_shares(_share_history()) == 49.0


def test_current_shares_ignore_a_filing_made_on_as_of() -> None:
    assert current_shares(pit_filter(_share_history(), date(2024, 5, 1))) == 52.0


def test_current_shares_are_chosen_by_filing_date_not_cover_date() -> None:
    visible = pd.concat(
        [
            _share_history(),
            _frame(_fact(SHARES, "2024-02-01", 47.0, filed="2024-06-01", form="10-K/A")),
        ],
        ignore_index=True,
    )
    assert current_shares(visible) == 47.0


def test_prior_shares_come_from_the_filing_that_first_reported_prior_end() -> None:
    visible = _share_history()
    assert prior_shares(visible, pd.Timestamp("2023-03-31")) == 52.0
    versioned = select_versions(visible)
    discarded = versioned.loc[versioned["end"] == pd.Timestamp("2023-03-31"), "accn"]
    assert discarded.tolist() == ["q1-24"]  # version selection alone would lose q1-23


def test_shares_are_nan_without_a_dei_fact() -> None:
    visible = _frame(_fact("Revenues", "2024-03-31", 110.0, start="2024-01-01"))
    assert math.isnan(current_shares(visible))
    assert math.isnan(prior_shares(visible, pd.Timestamp("2024-03-31")))


def test_prior_shares_are_nan_when_the_reporting_filing_has_no_cover_count() -> None:
    visible = _frame(
        _fact(
            "Revenues", "2023-03-31", 100.0, start="2023-01-01", filed="2023-05-01", accn="q1-23"
        ),
        _fact(SHARES, "2023-02-01", 53.0, filed="2023-02-15", form="10-K", accn="k22"),
    )
    assert math.isnan(prior_shares(visible, pd.Timestamp("2023-03-31")))


def test_prior_shares_are_nan_when_no_filing_reported_the_period() -> None:
    assert math.isnan(prior_shares(_share_history(), pd.Timestamp("2019-01-01")))


@pytest.mark.parametrize("reverse", [False, True])
def test_current_shares_same_day_filings_go_to_the_greater_accession(reverse: bool) -> None:
    rows = [
        _fact(SHARES, "2024-04-25", 49.0, accn="0001-24-000002"),
        _fact(SHARES, "2024-04-25", 48.0, accn="0001-24-000001"),
    ]
    visible = _frame(*(reversed(rows) if reverse else rows))
    assert current_shares(visible) == 49.0


@pytest.mark.parametrize("reverse", [False, True])
def test_prior_shares_first_reporter_and_cover_date_ignore_row_order(reverse: bool) -> None:
    rows = [
        _fact("Assets", "2023-03-31", 1.0, filed="2023-05-01", accn="0001-23-000002"),
        _fact(SHARES, "2023-04-25", 60.0, filed="2023-05-01", accn="0001-23-000002"),
        _fact("Assets", "2023-03-31", 1.0, filed="2023-05-01", accn="0001-23-000001"),
        _fact(SHARES, "2023-04-20", 51.0, filed="2023-05-01", accn="0001-23-000001"),
        _fact(SHARES, "2023-04-25", 52.0, filed="2023-05-01", accn="0001-23-000001"),
    ]
    visible = _frame(*(reversed(rows) if reverse else rows))
    assert prior_shares(visible, pd.Timestamp("2023-03-31")) == 52.0


@pytest.mark.parametrize("reverse", [False, True])
def test_shares_use_the_later_cover_date_of_the_chosen_filing(reverse: bool) -> None:
    rows = [
        _fact("Assets", "2024-03-31", 1.0, accn="q1-24"),
        _fact(SHARES, "2024-04-25", 49.0, accn="q1-24"),
        _fact(SHARES, "2024-04-20", 47.0, accn="q1-24"),
    ]
    visible = _frame(*(reversed(rows) if reverse else rows))
    assert current_shares(visible) == 49.0
    assert prior_shares(visible, pd.Timestamp("2024-03-31")) == 49.0


@pytest.mark.parametrize("reverse", [False, True])
def test_shares_are_nan_when_the_filing_has_distinct_counts_at_the_cover_date(
    reverse: bool,
) -> None:
    rows = [
        _fact("Assets", "2024-03-31", 1.0, accn="q1-24"),
        _fact(SHARES, "2024-04-25", 40.0, accn="q1-24"),  # e.g. class A leaking in undimensioned
        _fact(SHARES, "2024-04-25", 9.0, accn="q1-24"),  # class B
        _fact(SHARES, "2024-04-20", 49.0, accn="q1-24"),  # earlier cover date: not a fallback
    ]
    visible = _frame(*(reversed(rows) if reverse else rows))
    assert math.isnan(current_shares(visible))
    assert math.isnan(prior_shares(visible, pd.Timestamp("2024-03-31")))


def test_shares_identical_duplicate_counts_are_not_ambiguous() -> None:
    visible = _frame(
        _fact("Assets", "2024-03-31", 1.0, accn="q1-24"),
        _fact(SHARES, "2024-04-25", 49.0, accn="q1-24"),
        _fact(SHARES, "2024-04-25", 49.0, accn="q1-24"),
    )
    assert current_shares(visible) == 49.0
    assert prior_shares(visible, pd.Timestamp("2024-03-31")) == 49.0


def test_us_gaap_shares_fallback_at_instant() -> None:
    tags = [tag for tag in _spec("shares_outstanding").tags if tag != SHARES]
    facts = _frame(
        _fact("CommonStockSharesOutstanding", "2024-03-31", 48.0),
        _fact("CommonStockSharesOutstanding", "2023-03-31", 51.0, filed="2023-05-01"),
    )
    assert first_complete(facts, tags, pd.Timestamp("2024-03-31"), stock_value) == 48.0
    assert first_complete(facts, tags, pd.Timestamp("2023-03-31"), stock_value) == 51.0
    assert math.isnan(first_complete(facts, tags, pd.Timestamp("2022-03-31"), stock_value))


AS_OF = date(2024, 6, 1)


def _history(ticker: str) -> list[dict[str, Any]]:
    """Calendar-year filer: 10-K FY2022, 10-Q Q1 2023, 10-K FY2023, 10-Q Q1 2024."""

    def fact(
        tag: str, end: str, value: float, filing: str, filed: str, **kw: Any
    ) -> dict[str, Any]:
        form = "10-K" if filing.startswith("k") else "10-Q"
        accn = f"{ticker}-{filing}"
        return _fact(tag, end, value, filed=filed, form=form, accn=accn, ticker=ticker, **kw)

    return [
        fact("Revenues", "2022-12-31", 360.0, "k22", "2023-02-15", start="2022-01-01"),
        fact("Assets", "2022-12-31", 900.0, "k22", "2023-02-15"),
        fact(SHARES, "2023-02-01", 53.0, "k22", "2023-02-15"),
        fact("Revenues", "2023-03-31", 100.0, "q123", "2023-05-01", start="2023-01-01"),
        fact("Revenues", "2022-03-31", 90.0, "q123", "2023-05-01", start="2022-01-01"),
        fact("Assets", "2023-03-31", 950.0, "q123", "2023-05-01"),
        fact(SHARES, "2023-04-25", 52.0, "q123", "2023-05-01"),
        fact("Revenues", "2023-12-31", 400.0, "k23", "2024-02-15", start="2023-01-01"),
        fact("Assets", "2023-12-31", 1000.0, "k23", "2024-02-15"),
        fact(SHARES, "2024-02-01", 50.0, "k23", "2024-02-15"),
        fact("Revenues", "2024-03-31", 110.0, "q124", "2024-05-01", start="2024-01-01"),
        fact("Revenues", "2023-03-31", 100.0, "q124", "2024-05-01", start="2023-01-01"),
        fact("Assets", "2024-03-31", 1050.0, "q124", "2024-05-01"),
        fact(SHARES, "2024-04-25", 49.0, "q124", "2024-05-01"),
    ]


def test_derive_fundamentals_computes_current_and_prior_metrics() -> None:
    derived = derive_fundamentals(_frame(*_history("AAA")), ["AAA"], AS_OF)
    current, prior, periods = derived.current.loc["AAA"], derived.prior.loc["AAA"], derived.periods
    assert current["revenue"] == pytest.approx(410.0)  # 110 + 400 - 100
    assert prior["revenue"] == pytest.approx(370.0)  # 100 + 360 - 90
    assert current["total_assets"] == 1050.0
    assert prior["total_assets"] == 950.0
    assert current["shares_outstanding"] == 49.0
    assert prior["shares_outstanding"] == 52.0
    assert math.isnan(current["cash"])  # a missing metric leaves the ticker resolved
    assert periods.loc["AAA"].tolist() == [
        pd.Timestamp("2024-03-31"),
        pd.Timestamp("2023-03-31"),
        pd.Timestamp("2024-05-01"),
    ]
    assert derived.unresolved == {}


def test_dei_shares_win_over_us_gaap_and_us_gaap_is_the_fallback() -> None:
    both = [*_history("AAA"), _fact("CommonStockSharesOutstanding", "2024-03-31", 47.0)]
    gaap_only = [row for row in _history("BBB") if row["tag"] != SHARES] + [
        _fact("CommonStockSharesOutstanding", "2024-03-31", 48.0, ticker="BBB"),
    ]
    derived = derive_fundamentals(_frame(*both, *gaap_only), ["AAA", "BBB"], AS_OF)
    assert derived.current.loc["AAA", "shares_outstanding"] == 49.0
    assert derived.current.loc["BBB", "shares_outstanding"] == 48.0
    assert math.isnan(derived.prior.loc["BBB", "shares_outstanding"])


def test_ambiguous_dei_shares_fall_back_to_us_gaap() -> None:
    facts = _frame(
        *_history("AAA"),
        _fact(SHARES, "2024-04-25", 9.0, accn="AAA-q124"),  # a second class, undimensioned
        _fact("CommonStockSharesOutstanding", "2024-03-31", 47.0, accn="AAA-q124"),
    )
    derived = derive_fundamentals(facts, ["AAA"], AS_OF)
    assert derived.current.loc["AAA", "shares_outstanding"] == 47.0


def test_frames_are_sorted_typed_and_unresolved_is_reported() -> None:
    near_only = [
        row
        for row in _history("CCC")
        if row["accn"] == "CCC-q124" and row["tag"] != "us-gaap:Revenues"
    ]
    facts = pd.concat([_frame(*_history("AAA")), _frame(*near_only)])  # duplicate index labels
    derived = derive_fundamentals(facts, ["ZZZ", "CCC", "AAA", "AAA"], AS_OF)
    assert list(derived.current.index) == ["AAA", "CCC"]
    assert list(derived.prior.index) == ["AAA", "CCC"]
    assert list(derived.periods.index) == ["AAA", "CCC"]
    assert list(derived.current.columns) == list(METRIC_NAMES)
    assert list(derived.prior.columns) == list(METRIC_NAMES)
    assert list(derived.periods.columns) == list(PERIOD_COLUMNS)
    assert (derived.current.dtypes == "float64").all()
    assert (derived.prior.dtypes == "float64").all()
    assert (derived.periods.dtypes == "datetime64[ns]").all()
    assert dict(derived.unresolved) == {"ZZZ": NO_FILINGS}
    assert derived.prior.loc["CCC"].isna().all()
    assert pd.isna(derived.periods.loc["CCC", "prior_end"])
    assert derived.current.loc["CCC", "total_assets"] == 1050.0


def test_prior_year_us_gaap_shares_are_the_fallback_when_dei_is_absent_at_e_prime() -> None:
    rows = [
        row for row in _history("AAA") if not (row["tag"] == SHARES and row["accn"] == "AAA-q123")
    ]
    rows.append(_fact("CommonStockSharesOutstanding", "2023-03-31", 51.0, accn="AAA-q123"))
    derived = derive_fundamentals(_frame(*rows), ["AAA"], AS_OF)
    assert derived.prior.loc["AAA", "shares_outstanding"] == 51.0
    assert derived.current.loc["AAA", "shares_outstanding"] == 49.0


def test_prior_shares_see_the_filing_that_version_selection_discards() -> None:
    # Without the Q1 2023 Assets fact, only Revenues ends at E' and q124 restates it, so the
    # version-selected frame would credit E' to q124 (49) instead of its first reporter (52).
    rows = [
        row
        for row in _history("AAA")
        if not (row["tag"] == "us-gaap:Assets" and row["end"] == "2023-03-31")
    ]
    derived = derive_fundamentals(_frame(*rows), ["AAA"], AS_OF)
    assert derived.prior.loc["AAA", "shares_outstanding"] == 52.0


def test_values_use_the_latest_visible_restatement() -> None:
    restated = _fact("Assets", "2024-03-31", 1060.0, filed="2024-05-15", form="10-Q/A")
    derived = derive_fundamentals(_frame(*_history("AAA"), restated), ["AAA"], AS_OF)
    assert derived.current.loc["AAA", "total_assets"] == 1060.0
    assert derived.periods.loc["AAA", "last_filed"] == pd.Timestamp("2024-05-15")


def test_last_filed_ignores_later_filings_and_uncurated_tags() -> None:
    facts = _frame(
        *_history("AAA"),
        _fact("Assets", "2024-06-30", 1100.0, filed="2024-08-01"),
        _fact("NotACuratedTag", "2024-03-31", 5.0, filed="2024-05-20"),
    )
    derived = derive_fundamentals(facts, ["AAA"], AS_OF)
    assert derived.periods.loc["AAA"].tolist() == [
        pd.Timestamp("2024-03-31"),
        pd.Timestamp("2023-03-31"),
        pd.Timestamp("2024-05-01"),
    ]


def test_no_visible_filing_is_no_reference_period() -> None:
    derived = derive_fundamentals(_frame(*_history("AAA")), ["AAA"], date(2023, 2, 15))
    assert dict(derived.unresolved) == {"AAA": NO_REFERENCE_PERIOD}
    assert derived.current.empty


def test_dei_only_filings_are_no_reference_period() -> None:
    facts = _frame(_fact(SHARES, "2024-04-25", 49.0))
    assert dict(derive_fundamentals(facts, ["AAA"], AS_OF).unresolved) == {
        "AAA": NO_REFERENCE_PERIOD
    }


def test_staleness_boundary_keeps_exactly_max_days_and_drops_one_more() -> None:
    facts = _frame(*_history("AAA"))
    boundary = date(2024, 3, 31) + timedelta(days=MAX_STALENESS_DAYS)
    kept = derive_fundamentals(facts, ["AAA"], boundary)
    stale = derive_fundamentals(facts, ["AAA"], boundary + timedelta(days=1))
    assert MAX_STALENESS_DAYS == 200
    assert list(kept.current.index) == ["AAA"]
    assert dict(kept.unresolved) == {}
    assert dict(stale.unresolved) == {"AAA": STALE_FUNDAMENTALS}
    assert stale.current.empty


def test_all_unresolved_gives_empty_frames_with_the_contract_columns() -> None:
    derived = derive_fundamentals(_frame(), ["AAA"], AS_OF)
    assert derived.current.empty
    assert derived.prior.empty
    assert list(derived.current.columns) == list(METRIC_NAMES)
    assert list(derived.prior.columns) == list(METRIC_NAMES)
    assert list(derived.periods.columns) == list(PERIOD_COLUMNS)
    assert dict(derived.unresolved) == {"AAA": NO_FILINGS}


def test_unresolved_is_sorted_by_ticker() -> None:
    derived = derive_fundamentals(_frame(*_history("AAA")), ["ZZZ", "AAA", "MMM"], AS_OF)
    assert list(derived.unresolved) == ["MMM", "ZZZ"]
