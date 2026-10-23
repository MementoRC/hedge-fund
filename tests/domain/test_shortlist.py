"""Tests for the shortlist domain types."""

import dataclasses
from datetime import date

import pytest

from hedge_fund.domain.shortlist import TRIMMED_CAP, Candidate, Shortlist, Side


def test_side_values_are_plain_strings() -> None:
    assert [str(side) for side in Side] == ["long", "short"]


def test_trimmed_cap_reason() -> None:
    assert TRIMMED_CAP == "cap"


def test_shortlist_types_are_frozen() -> None:
    candidate = Candidate(ticker="A", sector="Tech", side=Side.LONG, depth=1, composite=1.5)
    result = Shortlist(as_of=date(2026, 9, 30), candidates=(candidate,), trimmed={})
    with pytest.raises(dataclasses.FrozenInstanceError):
        candidate.depth = 2  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.candidates = ()  # type: ignore[misc]
