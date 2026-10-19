"""Tests for the verdict gate, conviction score and alpha."""

import dataclasses

import pytest

from hedge_fund.domain.research import NameResearch, Verdict
from hedge_fund.domain.shortlist import Candidate, Side
from hedge_fund.engine.portfolio.alpha import DEFAULT_ALPHA, AlphaParams, alphas, gate, score

NAN = float("nan")
INF = float("inf")


def _name(
    ticker: str = "A",
    *,
    side: Side = Side.LONG,
    verdict: Verdict = Verdict.CONFIRM,
    mean: float | None = 2.0,
    std: float | None = 0.0,
) -> NameResearch:
    candidate = Candidate(ticker=ticker, sector="Tech", side=side, depth=1, composite=1.0)
    return NameResearch(
        candidate=candidate,
        runs=(),
        valid=3,
        mean=mean,
        std=std,
        agreement=1.0,
        verdict=verdict,
    )


# --- params ----------------------------------------------------------------


def test_default_params() -> None:
    assert AlphaParams(ic=0.05) == DEFAULT_ALPHA


@pytest.mark.parametrize("value", [NAN, INF, -INF, True, "x"])
def test_rejects_non_finite_or_non_numeric_ic(value: object) -> None:
    params = dataclasses.replace(DEFAULT_ALPHA, ic=value)
    with pytest.raises(ValueError, match="ic must be a finite number"):
        alphas([], {}, params=params)


@pytest.mark.parametrize("value", [0, 0.0, -0.05])
def test_rejects_non_positive_ic(value: float) -> None:
    params = dataclasses.replace(DEFAULT_ALPHA, ic=value)
    with pytest.raises(ValueError, match="ic must be > 0"):
        alphas([], {}, params=params)


# --- gate ------------------------------------------------------------------


def test_gate_keeps_confirm_in_order_and_explains_the_rest() -> None:
    names = [
        _name("C"),
        _name("N", verdict=Verdict.NEUTRAL),
        _name("A"),
        _name("X", verdict=Verdict.CONTRADICT),
        _name("I", verdict=Verdict.INSUFFICIENT, mean=None, std=None),
    ]
    eligible, excluded = gate(names)
    assert [name.candidate.ticker for name in eligible] == ["C", "A"]
    assert excluded == {
        "N": "verdict:neutral",
        "X": "verdict:contradict",
        "I": "verdict:insufficient",
    }


def test_gate_on_empty_research() -> None:
    assert gate([]) == ((), {})


# --- score -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("mean", "std", "expected"),
    [
        (0.5, 0.0, 0.25),  # (0.5 / 2) / (1 + 0)
        (2.0, 0.0, 1.0),  # (2 / 2) / (1 + 0)
        (0.5, 1.0, 0.125),  # (0.5 / 2) / (1 + 1)
        (2.0, 1.0, 0.5),  # (2 / 2) / (1 + 1)
    ],
)
def test_score(mean: float, std: float, expected: float) -> None:
    assert score(_name(mean=mean, std=std)) == pytest.approx(expected)


@pytest.mark.parametrize(
    "name",
    [
        _name(verdict=Verdict.NEUTRAL),
        _name(mean=None),
        _name(std=None),
    ],
)
def test_score_needs_a_confirm_name_with_stats(name: NameResearch) -> None:
    with pytest.raises(ValueError, match="CONFIRM name with stats"):
        score(name)


# --- alpha -----------------------------------------------------------------


def test_alpha_sign_follows_side() -> None:
    names = [_name("L"), _name("S", side=Side.SHORT)]
    out = alphas(names, {"L": 0.2, "S": 0.2})
    # 0.05 * 0.2 * score 1.0 = 0.01, negated for the short
    assert out == {"L": pytest.approx(0.01), "S": pytest.approx(-0.01)}


def test_alpha_scales_with_vol_score_and_ic() -> None:
    names = [_name("A"), _name("B", mean=0.5, std=1.0)]
    out = alphas(names, {"A": 0.4, "B": 0.2}, params=AlphaParams(ic=0.1))
    # A: 0.1 * 0.4 * 1.0 = 0.04; B: 0.1 * 0.2 * 0.125 = 0.0025
    assert out == {"A": pytest.approx(0.04), "B": pytest.approx(0.0025)}


def test_alpha_accepts_int_ic() -> None:
    assert alphas([_name("A")], {"A": 0.2}, params=AlphaParams(ic=1)) == {"A": pytest.approx(0.2)}
