"""Tests for the factor risk model: betas, exposures, universe, regression, F, D, decompose."""

import numpy as np
import pandas as pd
import pytest

from hedge_fund.engine.risk.factor_model import betas, estimation_universe, exposures

NAN = float("nan")
SPY = [0.01, -0.02, 0.015, 0.005, -0.01, 0.02]


def _index(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(end="2026-09-30", periods=n)


def _frame(columns: dict[str, list[float]]) -> pd.DataFrame:
    n = len(next(iter(columns.values())))
    return pd.DataFrame(columns, index=_index(n))


def _series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=_index(len(values)))


def _with_nan(values: list[float], *rows: int) -> list[float]:
    return [NAN if row in rows else value for row, value in enumerate(values)]


# --- betas -----------------------------------------------------------------


def test_beta_of_scaled_series() -> None:
    returns = _frame({"A": [2 * s for s in SPY], "B": [-0.5 * s for s in SPY]})
    assert betas(returns, _series(SPY)) == {"A": pytest.approx(2.0), "B": pytest.approx(-0.5)}


def test_beta_is_pairwise_per_name() -> None:
    # SPY has no return on row 5, A none on row 2, B none on row 3. B's pairwise rows are
    # {0, 1, 2, 4}; the rows shared by all names are {0, 1, 4} (no row 2). B is noisy, with
    # noise on row 2, so a shared-rows slope differs from B's pairwise slope.
    noise = [0.0, 0.02, 0.03, 0.0, -0.01, 0.0]
    b = [-s + n for s, n in zip(SPY, noise, strict=True)]
    returns = _frame({"A": _with_nan([2 * s for s in SPY], 2), "B": _with_nan(b, 3)})
    spy = _series(_with_nan(SPY, 5))

    def slope(rows: list[int]) -> float:
        x = np.array([b[r] for r in rows])
        s = np.array([SPY[r] for r in rows])
        return float(((x - x.mean()) @ (s - s.mean())) / ((s - s.mean()) @ (s - s.mean())))

    pairwise = slope([0, 1, 2, 4])
    assert abs(pairwise - slope([0, 1, 4])) > 1e-3  # the test discriminates
    out = betas(returns, spy)
    assert out["A"] == pytest.approx(2.0)
    assert out["B"] == pytest.approx(pairwise, rel=1e-12)


@pytest.mark.parametrize(
    "spy",
    [[0.01] * 6, _with_nan(SPY, 0, 1, 2)],
    ids=["flat", "no-shared-rows"],
)
def test_rejects_benchmark_without_variance_on_a_name_rows(spy: list[float]) -> None:
    returns = _frame({"A": _with_nan(SPY, 3, 4, 5)})
    with pytest.raises(ValueError, match="zero variance on 'A'"):
        betas(returns, _series(spy))


# --- exposures -------------------------------------------------------------


def test_exposures_are_beta_then_sorted_sector_dummies() -> None:
    sectors = {"A": "Tech", "B": "Energy", "C": "Tech", "Z": "Health"}  # Z: no beta, ignored
    out = exposures({"C": -0.3, "A": 1.2, "B": 0.5}, sectors)
    expected = pd.DataFrame(
        [[1.2, 0.0, 1.0], [0.5, 1.0, 0.0], [-0.3, 0.0, 1.0]],
        index=["A", "B", "C"],
        columns=["market", "Energy", "Tech"],
    )
    pd.testing.assert_frame_equal(out, expected)


# --- estimation universe ---------------------------------------------------

FULL = [0.01] * 6
SHORT_HISTORY = _with_nan(FULL, 0, 1, 2)  # 3 returns < min_obs 4


def test_universe_drops_short_history() -> None:
    returns = _frame({"A": FULL, "B": FULL, "C": SHORT_HISTORY})
    sectors = dict.fromkeys(["A", "B", "C"], "Tech")
    assert estimation_universe(returns, sectors, min_obs=4) == ["A", "B"]


def test_universe_drops_a_single_name_sector() -> None:
    returns = _frame({"A": FULL, "B": FULL, "S": FULL})
    sectors = {"A": "Tech", "B": "Tech", "S": "Solo"}
    assert estimation_universe(returns, sectors, min_obs=4) == ["A", "B"]


def test_universe_drops_a_sector_left_with_one_name_by_the_history_filter() -> None:
    returns = _frame({"A": FULL, "B": FULL, "E": FULL, "F": SHORT_HISTORY})
    sectors = {"A": "Tech", "B": "Tech", "E": "Energy", "F": "Energy"}
    assert estimation_universe(returns, sectors, min_obs=4) == ["A", "B"]


def test_universe_is_sorted_and_ignores_extra_sector_keys() -> None:
    returns = _frame({"D": FULL, "C": FULL, "B": FULL, "A": FULL})
    sectors = {"A": "Tech", "B": "Energy", "C": "Tech", "D": "Energy", "Q": "Solo"}
    assert estimation_universe(returns, sectors, min_obs=4) == ["A", "B", "C", "D"]
