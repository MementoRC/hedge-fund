"""store_coverage: the membership coverage of a frozen store over a run window."""

from __future__ import annotations

from datetime import date

import numpy as np

from hedge_fund.domain.backtest import FlaggedSpan, MemberSpan
from hedge_fund.services.backtest_coverage import store_coverage
from tests.factories import make_frozen_data

START, END = date(2024, 1, 1), date(2024, 2, 9)


def test_member_days_are_counted_against_the_stored_closes() -> None:
    report = store_coverage(make_frozen_data(), START, END)

    rows = {row.ticker: row for row in report.spans}
    assert (rows["AAA"].window_days, rows["AAA"].priced_days) == (30, 30)
    assert (rows["BBB"].window_days, rows["BBB"].priced_days) == (20, 20)
    assert (rows["CCC"].window_days, rows["CCC"].priced_days) == (10, 0)
    assert (report.member_days, report.priced_member_days) == (60, 50)
    assert (report.zero_spans, report.full_spans) == (1, 2)


def test_a_nan_close_is_not_priced() -> None:
    data = make_frozen_data()
    closes = data.closes.copy()
    closes.iloc[3, closes.columns.get_loc("AAA")] = np.nan

    report = store_coverage(make_frozen_data(closes=closes), START, END)

    aaa = next(row for row in report.spans if row.ticker == "AAA")
    assert aaa.priced_days == 29
    assert report.partial_spans == 1


def test_the_window_and_the_flagged_spans_bound_the_count() -> None:
    flagged = (FlaggedSpan(MemberSpan("CCC", date(2024, 1, 29), date(2024, 2, 2)), "Other"),)

    report = store_coverage(make_frozen_data(flagged=flagged), date(2024, 1, 22), date(2024, 2, 5))

    ccc = next(row for row in report.spans if row.ticker == "CCC")
    assert (report.start, report.end) == (date(2024, 1, 22), date(2024, 2, 5))
    # the flagged span is half-open: it holds Jan 29 to Feb 1
    assert (ccc.flagged_days, ccc.window_days) == (4, 2)
