"""Tests for the factor risk model: betas, exposures, universe, regression, F, D, decompose."""

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from hedge_fund.domain.risk import RiskBreakdown
from hedge_fund.engine.portfolio.risk_model import RiskParams, simple_returns
from hedge_fund.engine.risk.factor_model import (
    FactorModel,
    Regression,
    betas,
    covariances,
    decompose,
    empty_breakdown,
    estimation_universe,
    exposures,
    fit_factor_model,
    regress,
)
from tests.factories import make_closes

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


def test_universe_keeps_exactly_min_obs_returns_and_drops_one_fewer() -> None:
    exact = _with_nan(FULL, 0, 1)  # 4 returns == min_obs 4
    one_fewer = _with_nan(FULL, 0, 1, 2)  # 3 returns
    returns = _frame({"A": FULL, "B": exact, "C": one_fewer})
    sectors = dict.fromkeys(["A", "B", "C"], "Tech")
    assert estimation_universe(returns, sectors, min_obs=4) == ["A", "B"]


def test_universe_does_not_count_sector_names_missing_from_returns() -> None:
    returns = _frame({"A": FULL, "B": FULL, "E": FULL})
    sectors = {"A": "Tech", "B": "Tech", "E": "Energy", "X": "Energy"}  # X is not in returns
    assert estimation_universe(returns, sectors, min_obs=4) == ["A", "B"]


def test_universe_is_sorted_and_ignores_extra_sector_keys() -> None:
    returns = _frame({"D": FULL, "C": FULL, "B": FULL, "A": FULL})
    sectors = {"A": "Tech", "B": "Energy", "C": "Tech", "D": "Energy", "Q": "Solo"}
    assert estimation_universe(returns, sectors, min_obs=4) == ["A", "B", "C", "D"]


# --- regression ------------------------------------------------------------

# Tech: A, B share beta 1.0 and C has 2.0; Energy: D, E share 0.5 and G has 1.5.
X = exposures(
    {"A": 1.0, "B": 1.0, "C": 2.0, "D": 0.5, "E": 0.5, "G": 1.5},
    {"A": "Tech", "B": "Tech", "C": "Tech", "D": "Energy", "E": "Energy", "G": "Energy"},
)
# Factor returns per day as (market, Energy, Tech).
TRUE_F = np.array(
    [
        [0.010, 0.002, -0.003],
        [-0.004, 0.001, 0.002],
        [0.006, -0.002, 0.001],
        [0.003, 0.004, -0.001],
        [-0.008, 0.000, 0.003],
        [0.002, -0.001, -0.002],
    ]
)
# Day 1: E and G missing -> Energy has one name -> skipped.
# Day 2: C and G missing -> beta is constant within each sector, so the market column is a
#        combination of the dummies and rank(X_t) = 2 < 3 -> skipped.
# Day 3: G missing -> Tech betas still differ, rank 3 -> usable.
MISSING = {1: ["E", "G"], 2: ["C", "G"], 3: ["G"]}


def _exact_returns() -> pd.DataFrame:
    returns = pd.DataFrame(TRUE_F @ X.to_numpy().T, index=_index(6), columns=list(X.index))
    for row, tickers in MISSING.items():
        returns.iloc[row, [returns.columns.get_loc(t) for t in tickers]] = NAN
    return returns


def test_regression_skips_unusable_days_and_recovers_exact_factor_returns() -> None:
    out = regress(_exact_returns(), X, min_obs=4)
    assert list(out.factor_returns.index) == list(_index(6)[[0, 3, 4, 5]])
    assert list(out.factor_returns.columns) == ["market", "Energy", "Tech"]
    np.testing.assert_allclose(out.factor_returns.to_numpy(), TRUE_F[[0, 3, 4, 5]], atol=1e-12)
    residuals = out.residuals
    assert list(residuals.columns) == ["A", "B", "C", "D", "E", "G"]
    assert np.isnan(residuals.loc[_index(6)[3], "G"])
    np.testing.assert_allclose(residuals.drop(columns="G").to_numpy(), 0.0, atol=1e-12)


def test_regression_needs_min_obs_usable_days() -> None:
    with pytest.raises(ValueError, match="4 usable regression days, needs min_obs=5"):
        regress(_exact_returns(), X, min_obs=5)


def test_covariances_annualise_with_ddof_zero() -> None:
    # Factor returns: market [0.01, -0.01], Tech [0.02, 0.0]; both centre to +-0.01, so every
    # entry of F is 252 * 1e-4 = 0.0252. A's residuals [0.01, -0.01] give D = 0.0252; B has
    # one residual (< min_obs 2), so no estimate.
    index = _index(2)
    regression = Regression(
        factor_returns=pd.DataFrame({"market": [0.01, -0.01], "Tech": [0.02, 0.0]}, index=index),
        residuals=pd.DataFrame({"A": [0.01, -0.01], "B": [0.01, NAN]}, index=index),
    )
    factor_cov, specific = covariances(regression, min_obs=2)
    expected = pd.DataFrame(0.0252, index=["market", "Tech"], columns=["market", "Tech"])
    pd.testing.assert_frame_equal(factor_cov, expected)
    assert specific == {"A": pytest.approx(0.0252)}


def test_recovers_known_factor_and_specific_covariances() -> None:
    # 3 sectors x 30 names, 2000 days of r_t = X f_t + u_t with f_t ~ N(0, F / 252) and
    # u_t ~ N(0, diag(D) / 252), seeded, every day usable. Regressing on the true X,
    # f-hat = f + A u and u-hat = M u with A = (X'X)^-1 X' and M = I - X A, so
    #   E[F-hat] = F + A diag(D) A'        E[D-hat_i] = sum_j M_ij^2 D_j
    # (the leakage terms are a few percent here). What remains is sampling error: a variance
    # estimated from T = 2000 normal draws has relative standard error sqrt(2 / T) ~ 3.2%,
    # and an off-diagonal covariance about sqrt(F_ii F_jj / T) ~ 2.2% of sqrt(F_ii F_jj).
    # A 20% band on that scale is over 6 standard errors (no flakes, seed or not) yet fails
    # a missing x252, transposed exposures or residuals taken from the wrong fit.
    rng = np.random.default_rng(20261019)
    sectors = {f"{s}{i:02d}": s for s in ("Energy", "Health", "Tech") for i in range(30)}
    tickers = sorted(sectors)
    x = exposures(dict(zip(tickers, rng.uniform(0.5, 1.5, len(tickers)), strict=True)), sectors)
    true_f = np.array(
        [
            [0.04, 0.0, 0.003, 0.0],
            [0.0, 0.0225, 0.0, 0.0],
            [0.003, 0.0, 0.01, 0.0],
            [0.0, 0.0, 0.0, 0.0144],
        ]
    )  # market, Energy, Health, Tech
    true_d = rng.uniform(0.02, 0.09, len(tickers))
    days = 2000
    f = rng.multivariate_normal(np.zeros(4), true_f / 252, size=days)
    u = rng.normal(0.0, np.sqrt(true_d / 252), size=(days, len(tickers)))
    xa = x.to_numpy()
    returns = pd.DataFrame(f @ xa.T + u, index=_index(days), columns=tickers)
    factor_cov, specific = covariances(regress(returns, x, min_obs=days), min_obs=days)
    a = np.linalg.solve(xa.T @ xa, xa.T)
    m = np.eye(len(tickers)) - xa @ a
    expected_f = true_f + a @ np.diag(true_d) @ a.T
    expected_d = (m**2) @ true_d
    scale = np.sqrt(np.outer(np.diag(expected_f), np.diag(expected_f)))
    assert np.all(np.abs(factor_cov.to_numpy() - expected_f) <= 0.2 * scale)
    np.testing.assert_allclose([specific[t] for t in tickers], expected_d, rtol=0.2)


# --- fit_factor_model ------------------------------------------------------

UNIVERSE = {
    "AAA": "Tech",
    "ABB": "Tech",
    "ACC": "Tech",
    "BAA": "Health",
    "BBB": "Health",
    "BCC": "Health",
    "CAA": "Energy",
    "CBB": "Energy",
}
FIT = RiskParams(lookback=60, min_obs=40)


def _fit_inputs(
    extra: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, str]]:
    sectors = UNIVERSE | (extra or {})
    closes = make_closes(list(sectors), n_days=80, seed=5)
    spy = make_closes(["SPY"], n_days=80, seed=9)["SPY"]
    return closes, spy, sectors


def test_fit_uses_the_last_lookback_returns_of_every_name() -> None:
    closes, spy, sectors = _fit_inputs()
    model = fit_factor_model(closes, spy, sectors, params=FIT)
    assert model.usable_days == 60
    assert list(model.exposures.columns) == ["market", "Energy", "Health", "Tech"]
    assert list(model.exposures.index) == sorted(UNIVERSE)
    assert set(model.specific_var) == set(UNIVERSE)
    window = simple_returns(closes).iloc[-60:]
    spy_window = simple_returns(spy).iloc[-60:]
    assert model.exposures["market"].to_dict() == pytest.approx(betas(window, spy_window))
    np.testing.assert_allclose(model.factor_cov.to_numpy(), model.factor_cov.to_numpy().T)


def test_fit_ignores_benchmark_dates_absent_from_the_snapshot() -> None:
    closes, spy, sectors = _fit_inputs()
    weekend = pd.Series([1000.0], index=[pd.Timestamp("2026-09-27")])  # a Sunday
    noisy = pd.concat([spy, weekend]).sort_index()
    out = fit_factor_model(closes, noisy, sectors, params=FIT)
    pd.testing.assert_frame_equal(
        out.exposures, fit_factor_model(closes, spy, sectors, params=FIT).exposures
    )


def test_fit_rejects_invalid_params() -> None:
    closes, spy, sectors = _fit_inputs()
    with pytest.raises(ValueError, match="2 <= min_obs <= lookback"):
        fit_factor_model(closes, spy, sectors, params=RiskParams(lookback=1, min_obs=1))


# --- decompose -------------------------------------------------------------

FACTORS = ["market", "Energy", "Tech"]
HAND = FactorModel(
    exposures=pd.DataFrame(
        [
            [1.0, 0.0, 1.0],
            [0.5, 0.0, 1.0],
            [1.5, 0.0, 1.0],
            [-0.5, 1.0, 0.0],
            [0.5, 1.0, 0.0],
        ],
        index=["A", "B", "C", "E1", "E2"],
        columns=FACTORS,
    ),
    factor_cov=pd.DataFrame(
        [[0.04, 0.0, 0.005], [0.0, 0.01, 0.0], [0.005, 0.0, 0.02]],
        index=FACTORS,
        columns=FACTORS,
    ),
    specific_var={"A": 0.04, "B": 0.09, "C": 0.0625, "E1": 0.01, "E2": 0.0225},
    usable_days=252,
)


def _assert_additive(out: RiskBreakdown) -> None:
    assert out.factor_vol**2 + out.specific_vol**2 == pytest.approx(out.total_vol**2, rel=1e-10)
    assert sum(out.by_factor.values()) == pytest.approx(out.factor_vol**2, rel=1e-10)
    assert sum(out.by_name.values()) == pytest.approx(1.0, rel=1e-10)


def test_single_name_book_by_hand() -> None:
    # w_A = 0.1: x = X'w = (0.1, 0, 0.1); Fx = (0.0045, 0, 0.0025);
    # factor_var = 0.1 * 0.0045 + 0.1 * 0.0025 = 0.0007; specific = 0.01 * 0.04 = 0.0004.
    out = decompose(HAND, {"A": 0.1})
    assert out.total_vol == pytest.approx(math.sqrt(0.0011))
    assert out.factor_vol == pytest.approx(math.sqrt(0.0007))
    assert out.specific_vol == pytest.approx(math.sqrt(0.0004))
    assert out.factor_share == pytest.approx(7 / 11)
    assert out.by_factor == pytest.approx({"market": 0.00045, "Energy": 0.0, "Tech": 0.00025})
    assert out.by_name == pytest.approx({"A": 1.0})
    assert out.unassessed == ()


def test_mixed_book_is_additive() -> None:
    out = decompose(HAND, {"A": 0.1, "B": -0.05, "C": 0.03, "E1": -0.08, "E2": 0.06})
    _assert_additive(out)
    assert list(out.by_name) == ["A", "B", "C", "E1", "E2"]
    assert 0.0 < out.factor_share < 1.0


def test_specific_only_book_has_no_factor_risk() -> None:
    # Tech weights (-0.1, 0.05, 0.05) on betas (1.0, 0.5, 1.5): sector net 0 and beta
    # exposure -0.1 + 0.025 + 0.075 = 0, so x = 0. Specific: 0.01 * 0.04 + 0.0025 * 0.09 +
    # 0.0025 * 0.0625 = 0.00078125.
    out = decompose(HAND, {"A": -0.1, "B": 0.05, "C": 0.05})
    assert out.factor_share == pytest.approx(0.0, abs=1e-12)
    assert out.total_vol == pytest.approx(math.sqrt(0.00078125))
    assert sum(out.by_name.values()) == pytest.approx(1.0, rel=1e-10)


def test_pure_sector_bet_loads_on_that_sector() -> None:
    # E1 and E2 long 0.1 each: betas -0.5 and 0.5 cancel, so x = (0, 0.2, 0) and all factor
    # variance is Energy's: 0.2 * 0.01 * 0.2 = 0.0004.
    out = decompose(HAND, {"E1": 0.1, "E2": 0.1})
    assert out.by_factor == pytest.approx({"market": 0.0, "Energy": 0.0004, "Tech": 0.0})
    assert out.factor_vol**2 == pytest.approx(0.0004)
    _assert_additive(out)


def test_unassessable_names_are_left_out_and_listed() -> None:
    out = decompose(HAND, {"A": 0.1, "Q": 0.05, "P": -0.02})
    assert out.unassessed == ("P", "Q")
    assert dataclasses.replace(out, unassessed=()) == decompose(HAND, {"A": 0.1})


def test_empty_and_fully_unassessed_books_are_all_zero() -> None:
    zero = RiskBreakdown(0.0, 0.0, 0.0, 0.0, {}, {}, ())
    assert empty_breakdown() == zero
    assert decompose(HAND, {}) == zero
    assert decompose(HAND, {"Q": 0.1}) == dataclasses.replace(zero, unassessed=("Q",))


def test_zero_variance_book_has_zero_shares() -> None:
    flat = FactorModel(
        exposures=pd.DataFrame(
            [[0.0, 1.0], [0.0, 1.0]], index=["A", "B"], columns=["market", "Tech"]
        ),
        factor_cov=pd.DataFrame(0.0, index=["market", "Tech"], columns=["market", "Tech"]),
        specific_var={"A": 0.0, "B": 0.0},
        usable_days=1,
    )
    out = decompose(flat, {"A": 0.1, "B": -0.1})
    assert (out.total_vol, out.factor_share) == (0.0, 0.0)
    assert out.by_name == {"A": 0.0, "B": 0.0}


def test_fitted_model_unassessable_names() -> None:
    # SOLO is the only name in its sector (dropped); THIN lacks history; GONE is not in
    # the snapshot. All three are unassessable; the rest of the book is decomposed.
    closes, spy, sectors = _fit_inputs({"SOLO": "Solo", "THIN": "Tech"})
    closes.iloc[:50, closes.columns.get_loc("THIN")] = NAN  # 29 returns left < 40
    model = fit_factor_model(closes, spy, sectors, params=FIT)
    assert "SOLO" not in model.exposures.index
    assert "THIN" not in model.exposures.index
    out = decompose(model, {"AAA": 0.1, "CAA": -0.1, "SOLO": 0.05, "THIN": 0.05, "GONE": 0.01})
    assert out.unassessed == ("GONE", "SOLO", "THIN")
    assert list(out.by_name) == ["AAA", "CAA"]
    _assert_additive(out)


def test_fitted_model_full_book_is_additive() -> None:
    closes, spy, sectors = _fit_inputs()
    model = fit_factor_model(closes, spy, sectors, params=FIT)
    weights = {t: 0.1 if i % 2 == 0 else -0.1 for i, t in enumerate(sorted(UNIVERSE))}
    out = decompose(model, weights)
    _assert_additive(out)
    assert out.unassessed == ()
    assert all(math.isfinite(v) for v in (out.total_vol, out.factor_share))
