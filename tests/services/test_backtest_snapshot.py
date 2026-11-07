"""build_frozen_data: every ever-member plus SPY, with the unpriced and gap lists recorded."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from fractions import Fraction

import pandas as pd
import pytest

from hedge_fund.domain.backtest import Dividend, FlaggedSpan, FrozenData, MemberSpan
from hedge_fund.domain.ledger import Split
from hedge_fund.ports.fundamentals import FundamentalsProvider
from hedge_fund.services.backtest_snapshot import SnapshotSources, build_frozen_data
from tests.factories import make_facts
from tests.fakes import FakeFundamentalsProvider

START = date(2024, 1, 1)
END = date(2024, 1, 12)
DAYS = pd.bdate_range(START, END)  # 10 business days
SPANS = (
    MemberSpan("AAA", date(2020, 1, 2), None),
    MemberSpan("BBB", date(2020, 1, 2), date(2024, 1, 8)),
    MemberSpan("OLD", date(2020, 1, 2), date(2023, 6, 1)),
    MemberSpan("GONE", date(2023, 12, 1), None),
    MemberSpan("NEW", date(2024, 1, 8), None),
    MemberSpan("ZZZ", date(2024, 1, 2), None),
)
SECTORS = {"AAA": "Energy", "BBB": "Financials"}
CIKS = {"AAA": "0000000001", "BBB": "0000000002"}
COMPANY = {"NEW": "0000000003", "GONE": "0000000004"}
SICS = {"0000000003": "2834"}
SPLITS = (
    Split("AAA", date(2024, 1, 10), Fraction(2)),
    Split("GONE", date(2024, 1, 5), Fraction(3)),
    Split("AAA", date(2025, 3, 3), Fraction(4)),  # after END: kept, prices are adjusted for it
)
INSIDE = FlaggedSpan(MemberSpan("BBB", date(2023, 6, 1), date(2024, 1, 3)), "Other Corp")
OUTSIDE = FlaggedSpan(MemberSpan("OLD", date(2010, 1, 4), date(2012, 1, 3)), "Older Corp")


def _frame(close: float, days: pd.DatetimeIndex = DAYS) -> pd.DataFrame:
    return pd.DataFrame({"Open": close - 1.0, "Close": close, "Dividends": 0.0}, index=days)


def _histories() -> dict[str, pd.DataFrame | None]:
    aaa = _frame(50.0)
    aaa.loc[pd.Timestamp("2024-01-05"), "Dividends"] = 0.25
    return {
        "AAA": aaa,
        "BBB": _frame(20.0),
        "GONE": None,
        "NEW": _frame(30.0),
        "ZZZ": _frame(10.0),
        "SPY": _frame(470.0),
    }


class Vendors:
    """Fake vendors behind SnapshotSources, recording what each was asked."""

    def __init__(self, histories: Mapping[str, pd.DataFrame | None]) -> None:
        self.histories = dict(histories)
        self.history_calls: list[str] = []
        self.split_calls: list[tuple[str, ...]] = []
        self.company_calls = 0
        self.sic_calls: list[str] = []
        self.fundamentals_ciks: list[dict[str, str]] = []

    def sources(self) -> SnapshotSources:
        return SnapshotSources(
            history=self._history,
            splits=self._splits,
            company_ciks=self._company,
            sic=self._sic,
            fundamentals=self._fundamentals,
        )

    def _history(self, ticker: str, start: date, end: date) -> pd.DataFrame | None:
        self.history_calls.append(ticker)
        return self.histories.get(ticker)

    def _splits(self, tickers: Sequence[str]) -> tuple[Split, ...]:
        self.split_calls.append(tuple(tickers))
        return SPLITS

    def _company(self) -> Mapping[str, str]:
        self.company_calls += 1
        return COMPANY

    def _sic(self, cik: str) -> str | None:
        self.sic_calls.append(cik)
        return SICS.get(cik)

    def _fundamentals(self, ciks: Mapping[str, str]) -> FundamentalsProvider:
        self.fundamentals_ciks.append(dict(ciks))
        return FakeFundamentalsProvider(make_facts(list(ciks)))


def _build(
    vendors: Vendors,
    spans: Sequence[MemberSpan] = SPANS,
    flagged: Sequence[FlaggedSpan] = (),
) -> FrozenData:
    return build_frozen_data(
        vendors.sources(),
        spans,
        sectors=SECTORS,
        ciks=CIKS,
        start=START,
        end=END,
        flagged=flagged,
    )


def test_every_member_in_the_window_plus_spy_is_fetched_once() -> None:
    vendors = Vendors(_histories())
    _build(vendors)
    assert vendors.history_calls == ["AAA", "BBB", "GONE", "NEW", "ZZZ", "SPY"]


def test_prices_are_aligned_on_the_spy_calendar() -> None:
    data = _build(Vendors(_histories()))
    assert list(data.closes.columns) == ["AAA", "BBB", "NEW", "SPY", "ZZZ"]
    assert data.closes.index.equals(DAYS)
    assert data.opens["AAA"].iloc[0] == 49.0
    assert (data.start, data.end) == (START, END)
    assert data.membership == SPANS


def test_the_gaps_are_recorded() -> None:
    vendors = Vendors(_histories())
    data = _build(vendors)
    assert data.unpriced == ("GONE",)
    assert data.no_sector == ("GONE", "ZZZ")
    assert data.no_cik == ("ZZZ",)
    assert data.sic_sectors == ("NEW",)
    assert dict(data.sectors) == {"AAA": "Energy", "BBB": "Financials", "NEW": "Health Care"}
    assert vendors.sic_calls == ["0000000004", "0000000003"]


def test_ciks_fall_back_to_the_current_sec_map() -> None:
    data = _build(Vendors(_histories()))
    assert dict(data.ciks) == {
        "AAA": "0000000001",
        "BBB": "0000000002",
        "GONE": "0000000004",
        "NEW": "0000000003",
    }


def test_splits_keep_priced_tickers_and_later_events() -> None:
    vendors = Vendors(_histories())
    data = _build(vendors)
    assert vendors.split_calls == [("AAA", "BBB", "NEW", "SPY", "ZZZ")]
    assert data.splits == (SPLITS[0], SPLITS[2])


def test_dividends_come_from_the_history() -> None:
    assert _build(Vendors(_histories())).dividends == (Dividend("AAA", date(2024, 1, 5), 0.25),)


def test_facts_are_fetched_for_every_member_with_a_cik() -> None:
    vendors = Vendors(_histories())
    data = _build(vendors)
    assert vendors.fundamentals_ciks == [dict(data.ciks)]
    assert set(data.facts["ticker"]) == {"AAA", "BBB", "GONE", "NEW"}


def test_the_company_lookup_is_skipped_when_the_static_ciks_cover_every_member() -> None:
    vendors = Vendors(_histories())
    data = _build(vendors, spans=(SPANS[0],))
    assert vendors.company_calls == 0
    assert data.no_cik == ()


def test_without_any_cik_no_facts_are_fetched() -> None:
    vendors = Vendors(_histories())
    data = _build(vendors, spans=(SPANS[5],))
    assert vendors.fundamentals_ciks == []
    assert data.facts.empty


def test_rows_outside_the_window_or_off_the_spy_calendar_are_dropped() -> None:
    histories = _histories()
    histories["AAA"] = _frame(50.0, pd.bdate_range("2023-12-28", "2024-01-15"))
    histories["ZZZ"] = _frame(10.0, pd.DatetimeIndex(["2024-01-06"]))  # a Saturday
    data = _build(Vendors(histories))
    assert data.closes.index.equals(DAYS)
    assert "ZZZ" not in data.closes.columns
    assert data.unpriced == ("GONE", "ZZZ")


def test_an_all_nan_history_is_unpriced() -> None:
    histories = _histories()
    histories["BBB"] = _frame(float("nan"))
    assert _build(Vendors(histories)).unpriced == ("BBB", "GONE")


def test_spy_without_prices_is_refused() -> None:
    histories = _histories()
    histories["SPY"] = None
    with pytest.raises(ValueError, match="SPY has no prices"):
        _build(Vendors(histories))


def test_start_after_end_is_refused() -> None:
    with pytest.raises(ValueError, match="after end"):
        build_frozen_data(
            Vendors(_histories()).sources(), SPANS, sectors=SECTORS, ciks=CIKS, start=END, end=START
        )


def test_flagged_spans_default_to_none() -> None:
    assert _build(Vendors(_histories())).flagged == ()


def test_a_flagged_span_inside_the_window_is_kept_and_one_outside_is_dropped() -> None:
    data = _build(Vendors(_histories()), flagged=(OUTSIDE, INSIDE))
    assert data.flagged == (INSIDE,)
