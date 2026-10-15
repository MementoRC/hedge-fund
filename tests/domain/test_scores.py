"""Tests for scoring result domain types."""

import dataclasses
from datetime import date

import pytest

from hedge_fund.domain.scores import CrowdingReport, FactorCrowding, FactorScore, ScoreResult


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


def test_factor_score_quality_fields_default_to_none() -> None:
    s = FactorScore(
        ticker="T", sector="Tech", raw={}, z={}, composite=0.0, fallback_universe_z=False
    )
    assert s.piotroski is None
    assert s.altman_z is None


def test_factor_score_carries_quality_fields() -> None:
    s = FactorScore(
        ticker="T",
        sector="Tech",
        raw={},
        z={},
        composite=0.0,
        fallback_universe_z=False,
        piotroski=7,
        altman_z=2.5,
    )
    assert (s.piotroski, s.altman_z) == (7, 2.5)


def _factor_crowding() -> FactorCrowding:
    return FactorCrowding(
        factor="momentum_12_1",
        long_excess=0.25,
        short_excess=None,
        long_n=6,
        short_n=6,
        crowded=True,
    )


def test_crowding_report_holds_fields() -> None:
    report = CrowdingReport(
        as_of=date(2026, 1, 6), baseline_corr=-0.02, factors=(_factor_crowding(),)
    )
    assert report.as_of == date(2026, 1, 6)
    assert report.baseline_corr == -0.02
    factor = report.factors[0]
    assert factor.factor == "momentum_12_1"
    assert (factor.long_excess, factor.short_excess) == (0.25, None)
    assert (factor.long_n, factor.short_n, factor.crowded) == (6, 6, True)


def test_crowding_types_compare_by_value() -> None:
    assert _factor_crowding() == _factor_crowding()
    empty = CrowdingReport(as_of=date(2026, 1, 6), baseline_corr=None, factors=())
    assert empty == CrowdingReport(as_of=date(2026, 1, 6), baseline_corr=None, factors=())


def test_crowding_types_are_frozen() -> None:
    factor = _factor_crowding()
    with pytest.raises(dataclasses.FrozenInstanceError):
        factor.crowded = False  # type: ignore[misc]
    report = CrowdingReport(as_of=date(2026, 1, 6), baseline_corr=None, factors=())
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.baseline_corr = 0.1  # type: ignore[misc]
