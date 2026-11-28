"""reflag: a frozen store's flagged spans replaced by an edited list, nothing else touched."""

from __future__ import annotations

import dataclasses
from datetime import date

import pandas as pd

from hedge_fund.domain.backtest import FlaggedSpan, MemberSpan
from hedge_fund.services.backtest_reflag import reflag
from tests.factories import make_frozen_data

OLD = FlaggedSpan(MemberSpan("AAA", date(2023, 6, 1), date(2024, 1, 3)), "Old Corp")
KEPT = FlaggedSpan(MemberSpan("BBB", date(2023, 6, 1), date(2024, 1, 3)), "Kept Corp")
NEW = FlaggedSpan(MemberSpan("CCC", date(2023, 1, 2), date(2024, 1, 20)), "New Corp")
OUTSIDE = FlaggedSpan(MemberSpan("ZZZ", date(2010, 1, 4), date(2012, 1, 3)), "Older Corp")


def test_only_the_flagged_spans_are_replaced() -> None:
    data = make_frozen_data(flagged=(OLD, KEPT))

    result = reflag(data, (KEPT, NEW))

    assert result.data.flagged == (KEPT, NEW)
    for field in dataclasses.fields(data):
        if field.name == "flagged":
            continue
        before, after = getattr(data, field.name), getattr(result.data, field.name)
        if isinstance(before, pd.DataFrame):
            pd.testing.assert_frame_equal(after, before)
        else:
            assert after == before, field.name


def test_added_and_removed_are_the_differences() -> None:
    result = reflag(make_frozen_data(flagged=(OLD, KEPT)), (KEPT, NEW))

    assert result.added == (NEW,)
    assert result.removed == (OLD,)


def test_a_span_outside_the_store_window_is_dropped() -> None:
    result = reflag(make_frozen_data(), (OUTSIDE, KEPT))

    assert result.data.flagged == (KEPT,)
    assert result.added == (KEPT,)
    assert result.removed == ()
