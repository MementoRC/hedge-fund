"""Tests for the same-side correlated-pair pre-filter."""

import dataclasses

import pandas as pd
import pytest

from hedge_fund.domain.shortlist import Side
from hedge_fund.engine.portfolio.correlation import (
    DEFAULT_CORRELATION,
    CorrelationParams,
    correlation_filter,
)

NAN = float("nan")
INF = float("inf")
LONG = Side.LONG
SHORT = Side.SHORT


def _corr(tickers: list[str], pairs: dict[tuple[str, str], float]) -> pd.DataFrame:
    """Identity correlation with the given symmetric off-diagonal entries."""
    frame = pd.DataFrame(0.0, index=tickers, columns=tickers)
    for ticker in tickers:
        frame.loc[ticker, ticker] = 1.0
    for (a, b), rho in pairs.items():
        frame.loc[a, b] = rho
        frame.loc[b, a] = rho
    return frame


# --- params ----------------------------------------------------------------


def test_default_params() -> None:
    assert CorrelationParams(max_rho=0.85) == DEFAULT_CORRELATION


@pytest.mark.parametrize("value", [NAN, INF, True, "x"])
def test_rejects_non_finite_or_non_numeric_max_rho(value: object) -> None:
    params = dataclasses.replace(DEFAULT_CORRELATION, max_rho=value)
    with pytest.raises(ValueError, match="max_rho must be a finite number"):
        correlation_filter({}, {}, _corr([], {}), params=params)


@pytest.mark.parametrize("value", [-1.0, -1, -2.0, 1.01])
def test_rejects_max_rho_outside_range(value: float) -> None:
    params = dataclasses.replace(DEFAULT_CORRELATION, max_rho=value)
    with pytest.raises(ValueError, match="-1 < max_rho <= 1"):
        correlation_filter({}, {}, _corr([], {}), params=params)


# --- filter ----------------------------------------------------------------


def test_same_side_duplicate_is_dropped_naming_the_kept_ticker() -> None:
    corr = _corr(["A", "B"], {("A", "B"): 0.9})
    kept, excluded = correlation_filter({"A": 0.5, "B": 1.0}, {"A": LONG, "B": LONG}, corr)
    assert kept == ("B",)  # B scores higher, so it is kept first
    assert excluded == {"A": "correlated:B"}


def test_threshold_is_exclusive() -> None:
    corr = _corr(["A", "B"], {("A", "B"): 0.85})
    kept, excluded = correlation_filter({"A": 1.0, "B": 0.5}, {"A": LONG, "B": LONG}, corr)
    assert kept == ("A", "B")
    assert excluded == {}


def test_opposite_sides_are_never_filtered() -> None:
    corr = _corr(["A", "B"], {("A", "B"): 0.99})
    kept, excluded = correlation_filter({"A": 1.0, "B": 0.5}, {"A": LONG, "B": SHORT}, corr)
    assert kept == ("A", "B")
    assert excluded == {}


def test_ties_break_by_ticker_and_reason_names_first_clashing_kept_name() -> None:
    # Keep order: B (score 1.0), then A and D (0.5 each, ticker order), then C (0.25).
    # D clashes with B only; C clashes with A and B and is charged to B (kept first).
    # Drops happen in the order D, C; the result is sorted by ticker: C, D.
    corr = _corr(
        ["A", "B", "C", "D"],
        {("B", "D"): 0.9, ("A", "C"): 0.95, ("B", "C"): 0.9},
    )
    scores = {"C": 0.25, "D": 0.5, "B": 1.0, "A": 0.5}
    sides = dict.fromkeys(scores, SHORT)
    kept, excluded = correlation_filter(scores, sides, corr)
    assert kept == ("B", "A")
    assert excluded == {"C": "correlated:B", "D": "correlated:B"}
    assert list(excluded) == ["C", "D"]


def test_dropped_names_do_not_block_later_names() -> None:
    # B is dropped for A; C correlates only with B, so C stays.
    corr = _corr(["A", "B", "C"], {("A", "B"): 0.9, ("B", "C"): 0.9})
    scores = {"A": 1.0, "B": 0.75, "C": 0.5}
    kept, excluded = correlation_filter(scores, dict.fromkeys(scores, LONG), corr)
    assert kept == ("A", "C")
    assert excluded == {"B": "correlated:A"}


def test_nan_correlation_counts_as_uncorrelated() -> None:
    corr = _corr(["A", "Z"], {("A", "Z"): NAN})
    kept, excluded = correlation_filter({"A": 1.0, "Z": 0.5}, {"A": LONG, "Z": LONG}, corr)
    assert kept == ("A", "Z")
    assert excluded == {}


def test_max_rho_one_never_filters() -> None:
    corr = _corr(["A", "B"], {("A", "B"): 1.0})
    kept, _ = correlation_filter(
        {"A": 1.0, "B": 0.5},
        {"A": LONG, "B": LONG},
        corr,
        params=CorrelationParams(max_rho=1),
    )
    assert kept == ("A", "B")


def test_empty_input() -> None:
    assert correlation_filter({}, {}, _corr([], {})) == ((), {})
