"""Price coverage of the index membership history."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from hedge_fund.domain.backtest import FlaggedSpan, MemberSpan
from hedge_fund.engine.backtest.coverage import measure_coverage

D0 = date(2024, 1, 1)
CAL = tuple(D0 + timedelta(days=i) for i in range(10))  # D0 .. D0+9


def _d(i: int) -> date:
    return D0 + timedelta(days=i)


def _measure(spans, priced, flagged=(), first=0, last=9):
    return measure_coverage(spans, CAL, priced, flagged, _d(first), _d(last))


def test_a_fully_priced_span():
    report = _measure([MemberSpan("AAA", _d(0), _d(5))], {"AAA": set(CAL)})
    (row,) = report.spans
    assert (row.window_days, row.priced_days, row.flagged_days) == (5, 5, 0)
    assert row.first_priced == _d(0)
    assert row.coverage == 1.0
    assert (report.zero_spans, report.partial_spans, report.full_spans) == (0, 0, 1)
    assert report.coverage == 1.0


def test_a_ticker_absent_from_priced_has_zero_coverage():
    report = _measure([MemberSpan("AAA", _d(0), _d(4))], {})
    (row,) = report.spans
    assert (row.window_days, row.priced_days) == (4, 0)
    assert row.first_priced is None
    assert row.coverage == 0.0
    assert (report.zero_spans, report.partial_spans, report.full_spans) == (1, 0, 0)
    assert report.coverage == 0.0


def test_partial_coverage_and_first_priced():
    priced = {"AAA": {_d(2), _d(3)}}
    report = _measure([MemberSpan("AAA", _d(0), _d(4))], priced)
    (row,) = report.spans
    assert (row.window_days, row.priced_days) == (4, 2)
    assert row.first_priced == _d(2)
    assert row.coverage == 0.5
    assert (report.zero_spans, report.partial_spans, report.full_spans) == (0, 1, 0)


def test_span_end_is_exclusive():
    report = _measure([MemberSpan("AAA", _d(0), _d(3))], {"AAA": {_d(3)}})
    (row,) = report.spans
    assert (row.window_days, row.priced_days) == (3, 0)


def test_an_open_span_is_clipped_to_the_window():
    report = _measure([MemberSpan("AAA", _d(2), None)], {"AAA": set(CAL)}, last=6)
    (row,) = report.spans
    assert row.window_days == 5


def test_days_outside_the_window_are_ignored():
    report = _measure([MemberSpan("AAA", _d(0), _d(9))], {"AAA": {_d(0), _d(4)}}, first=3, last=5)
    (row,) = report.spans
    assert (row.window_days, row.priced_days, row.first_priced) == (3, 1, _d(4))
    assert (report.start, report.end) == (_d(3), _d(5))


def test_flagged_days_move_out_of_the_window_and_the_fraction():
    spans = [MemberSpan("AAA", _d(0), _d(6))]
    flagged = [FlaggedSpan(MemberSpan("AAA", _d(2), _d(4)), "other co")]
    report = _measure(spans, {"AAA": {_d(0), _d(1), _d(5)}}, flagged)
    (row,) = report.spans
    assert (row.window_days, row.flagged_days, row.priced_days) == (4, 2, 3)
    assert row.coverage == 0.75
    assert (report.member_days, report.priced_member_days, report.flagged_member_days) == (4, 3, 2)
    assert report.coverage == 0.75


def test_a_flag_for_another_ticker_is_ignored():
    flagged = [FlaggedSpan(MemberSpan("BBB", _d(0), _d(6)), "x")]
    report = _measure([MemberSpan("AAA", _d(0), _d(3))], {"AAA": set(CAL)}, flagged)
    assert report.spans[0].flagged_days == 0


def test_an_all_flagged_span_is_not_zero_coverage():
    spans = [MemberSpan("AAA", _d(0), _d(3))]
    flagged = [FlaggedSpan(MemberSpan("AAA", _d(0), _d(3)), "x")]
    report = _measure(spans, {}, flagged)
    (row,) = report.spans
    assert (row.window_days, row.flagged_days, row.coverage) == (0, 3, None)
    assert (report.zero_spans, report.partial_spans, report.full_spans) == (0, 0, 0)
    assert report.flagged_member_days == 3
    assert report.coverage is None


def test_a_span_outside_the_window_is_left_out():
    report = _measure([MemberSpan("AAA", _d(0), _d(2))], {"AAA": set(CAL)}, first=5)
    assert report.spans == ()


def test_two_spans_of_one_ticker_are_rows_sorted_by_start():
    spans = [MemberSpan("AAA", _d(6), None), MemberSpan("AAA", _d(0), _d(2))]
    report = _measure(spans, {"AAA": {_d(0), _d(1)}})
    assert [(r.start, r.priced_days, r.window_days) for r in report.spans] == [
        (_d(0), 2, 2),
        (_d(6), 0, 4),
    ]
    assert (report.zero_spans, report.full_spans) == (1, 1)
    assert report.coverage == pytest.approx(2 / 6)


def test_rows_sort_by_ticker_then_start():
    spans = [MemberSpan("BBB", _d(0), _d(2)), MemberSpan("AAA", _d(1), _d(3))]
    report = _measure(spans, {})
    assert [r.ticker for r in report.spans] == ["AAA", "BBB"]


def test_empty_inputs_have_zero_counts_and_no_fraction():
    report = measure_coverage([], [], {}, [], _d(0), _d(9))
    assert report.spans == ()
    assert (report.member_days, report.priced_member_days, report.flagged_member_days) == (0, 0, 0)
    assert (report.zero_spans, report.partial_spans, report.full_spans) == (0, 0, 0)
    assert report.coverage is None
