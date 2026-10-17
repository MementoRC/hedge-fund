"""Tests for the conviction engine."""

import dataclasses

import pytest

from hedge_fund.domain.shortlist import Candidate, Side
from hedge_fund.engine.research.conviction import (
    DEFAULT_CONVICTION,
    ConvictionParams,
    conviction,
)

NAN = float("nan")
INF = float("inf")


def _candidate(side: Side = Side.LONG) -> Candidate:
    return Candidate(ticker="A", sector="Tech", side=side, depth=1, composite=1.0)


# --- params ----------------------------------------------------------------


def test_default_params() -> None:
    assert DEFAULT_CONVICTION == ConvictionParams(
        min_valid=2, confirm=0.5, contradict=-0.5, max_std=1.0
    )


@pytest.mark.parametrize("value", [0, -1, 1.5, NAN, True])
def test_rejects_invalid_min_valid(value: object) -> None:
    params = dataclasses.replace(DEFAULT_CONVICTION, min_valid=value)
    with pytest.raises(ValueError, match="min_valid must be an int >= 1"):
        conviction(_candidate(), (), params=params)


@pytest.mark.parametrize("field", ["confirm", "contradict", "max_std"])
@pytest.mark.parametrize("value", [NAN, INF, -INF, True, "x"])
def test_rejects_non_finite_or_non_numeric(field: str, value: object) -> None:
    params = dataclasses.replace(DEFAULT_CONVICTION, **{field: value})
    with pytest.raises(ValueError, match=f"{field} must be a finite number"):
        conviction(_candidate(), (), params=params)


@pytest.mark.parametrize(
    "overrides",
    [{"confirm": 0.0}, {"confirm": -1.0}, {"contradict": 0.0}, {"contradict": 0.5}],
)
def test_rejects_misordered_thresholds(overrides: dict[str, float]) -> None:
    params = dataclasses.replace(DEFAULT_CONVICTION, **overrides)
    with pytest.raises(ValueError, match="contradict < 0 < confirm"):
        conviction(_candidate(), (), params=params)


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_rejects_non_positive_max_std(value: float) -> None:
    params = dataclasses.replace(DEFAULT_CONVICTION, max_std=value)
    with pytest.raises(ValueError, match="max_std must be > 0"):
        conviction(_candidate(), (), params=params)
