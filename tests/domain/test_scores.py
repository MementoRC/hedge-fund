"""Tests for scoring result domain types."""

import dataclasses
from datetime import date

import pytest

from hedge_fund.domain.scores import FactorScore, ScoreResult


def _factor_score() -> FactorScore:
    return FactorScore(
        ticker="AAA",
        sector="Tech",
        raw={"momentum_12_1": 0.1},
        z={"momentum_12_1": 1.0},
        composite=1.0,
        fallback_universe_z=False,
    )


def test_score_result_holds_fields() -> None:
    result = ScoreResult(
        as_of=date(2026, 1, 6),
        scores=(_factor_score(),),
        excluded={"BBB": "missing_data"},
    )
    assert result.as_of == date(2026, 1, 6)
    assert result.scores[0].ticker == "AAA"
    assert result.excluded == {"BBB": "missing_data"}


def test_results_compare_by_value() -> None:
    assert _factor_score() == _factor_score()


def test_factor_score_is_frozen() -> None:
    score = _factor_score()
    with pytest.raises(dataclasses.FrozenInstanceError):
        score.composite = 2.0  # type: ignore[misc]
