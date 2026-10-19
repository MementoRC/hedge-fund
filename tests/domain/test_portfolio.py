"""Tests for the portfolio domain types."""

import dataclasses
from datetime import date

import pytest

from hedge_fund.domain.portfolio import (
    EXCLUDED_HISTORY,
    EXCLUDED_INFEASIBLE,
    EXCLUDED_OPTIMIZER_ZERO,
    EXCLUDED_THIN_BOOK,
    Position,
    TargetPortfolio,
)
from hedge_fund.domain.shortlist import Side


def test_exclusion_reasons() -> None:
    assert [
        EXCLUDED_HISTORY,
        EXCLUDED_THIN_BOOK,
        EXCLUDED_OPTIMIZER_ZERO,
        EXCLUDED_INFEASIBLE,
    ] == ["history", "thin-book", "optimizer:zero", "infeasible"]


def test_portfolio_types_are_frozen() -> None:
    position = Position(
        ticker="A", sector="Tech", side=Side.SHORT, weight=-0.1, alpha=-0.01, beta=1.2
    )
    portfolio = TargetPortfolio(
        as_of=date(2026, 9, 30),
        positions=(position,),
        excluded={"B": "history"},
        gross=0.1,
        net=-0.1,
        beta=-0.12,
        expected_vol=0.02,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        position.weight = 0.2  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        portfolio.positions = ()  # type: ignore[misc]
