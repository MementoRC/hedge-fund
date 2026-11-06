"""Tests for the backtest domain types."""

from __future__ import annotations

from datetime import date

import pytest

from hedge_fund.domain.backtest import MemberSpan


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
