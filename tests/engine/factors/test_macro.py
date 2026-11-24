"""Macro-sensitivity factors: point-in-time OLS betas on daily FRED series changes."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date

import numpy as np
import pandas as pd
import pytest

from hedge_fund.domain.macro import MACRO_SERIES, RATE_SERIES, RISKOFF_SERIES
from hedge_fund.engine.factors.definitions import TRADING_DAYS
from hedge_fund.engine.factors.macro import (
    MACRO_FACTORS,
    MIN_PAIRS,
    MacroFactorSpec,
    macro_beta,
    macro_series_problem,
    rate_beta,
    riskoff_beta,
)
from tests.factories import END_DATE

DAYS = pd.bdate_range(end=END_DATE, periods=253)
AS_OF = DAYS[-1].date()
Factor = Callable[[pd.DataFrame, pd.DataFrame, date], pd.Series]


def _macro(days: pd.DatetimeIndex = DAYS, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(days)
    return pd.DataFrame(
        {
            RATE_SERIES: 4.0 + np.cumsum(rng.normal(0.0, 0.05, n)),
            RISKOFF_SERIES: 20.0 + np.cumsum(rng.normal(0.0, 0.8, n)),
        },
        index=days,
    )


def _closes(
    macro: pd.DataFrame,
    betas: dict[str, tuple[float, float]],
    *,
    lag: int = 0,
    noise: float = 1e-4,
    seed: int = 2,
) -> pd.DataFrame:
    """Closes whose log return on day d is b_rate * dDGS10(d - lag) + b_vix * dVIX(d - lag)
    plus N(0, noise)."""
    rng = np.random.default_rng(seed)
    changes = macro.diff().shift(lag).fillna(0.0)
    columns = {}
    for ticker, (b_rate, b_vix) in betas.items():
        returns = (
            b_rate * changes[RATE_SERIES]
            + b_vix * changes[RISKOFF_SERIES]
            + rng.normal(0.0, noise, len(macro))
        )
        columns[ticker] = 100.0 * np.exp(returns.cumsum())
    return pd.DataFrame(columns, index=macro.index)


def test_the_macro_factor_registry() -> None:
    assert MACRO_SERIES == ("DGS10", "VIXCLS")
    assert MIN_PAIRS == 200
    assert all(isinstance(spec, MacroFactorSpec) for spec in MACRO_FACTORS)
    assert [spec.name for spec in MACRO_FACTORS] == ["rate_beta", "riskoff_beta"]
    assert [spec.fn for spec in MACRO_FACTORS] == [rate_beta, riskoff_beta]
    assert [spec.sign for spec in MACRO_FACTORS] == [1, 1]
    assert [spec.min_history for spec in MACRO_FACTORS] == [TRADING_DAYS + 1] * 2


def test_known_betas_are_recovered_for_both_series() -> None:
    macro = _macro()
    closes = _closes(macro, {"R": (0.02, 0.0), "V": (0.0, -0.01)})
    assert macro_beta(closes, macro[RATE_SERIES], AS_OF)["R"] == pytest.approx(0.02, rel=0.05)
    assert macro_beta(closes, macro[RISKOFF_SERIES], AS_OF)["V"] == pytest.approx(-0.01, rel=0.05)


def test_rate_beta_penalises_positive_and_negative_rate_sensitivity_alike() -> None:
    macro = _macro()
    closes = _closes(macro, {"BANK": (0.02, 0.0), "UTIL": (-0.02, 0.0), "FLAT": (0.0, 0.0)})
    raw = rate_beta(closes, macro, AS_OF)
    assert raw["BANK"] == pytest.approx(-0.02, rel=0.05)
    assert raw["UTIL"] == pytest.approx(-0.02, rel=0.05)
    assert raw["FLAT"] > raw["BANK"]
    assert raw["FLAT"] > raw["UTIL"]
    assert (raw <= 0).all()


def test_riskoff_beta_ranks_a_name_that_holds_up_when_vix_jumps_higher() -> None:
    macro = _macro()
    closes = _closes(macro, {"HOLD": (0.0, -0.001), "CRASH": (0.0, -0.01)})
    raw = riskoff_beta(closes, macro, AS_OF)
    assert raw["CRASH"] == pytest.approx(-0.01, rel=0.05)
    assert raw["HOLD"] > raw["CRASH"]


@pytest.mark.parametrize("factor", [rate_beta, riskoff_beta])
def test_values_dated_as_of_have_no_effect(factor: Factor) -> None:
    macro = _macro()
    closes = _closes(macro, {"R": (0.02, -0.01)})
    moved_macro = macro.copy()
    moved_macro.iloc[-1] = [99.0, 99.0]
    moved_closes = closes.copy()
    moved_closes.iloc[-1] = closes.iloc[-1] * 1.5
    base = factor(closes, macro, AS_OF)
    pd.testing.assert_series_equal(factor(closes, moved_macro, AS_OF), base)
    pd.testing.assert_series_equal(factor(moved_closes, macro, AS_OF), base)


@pytest.mark.parametrize("factor", [rate_beta, riskoff_beta])
def test_on_a_holiday_as_of_the_last_close_counts_and_the_as_of_macro_row_does_not(
    factor: Factor,
) -> None:
    macro = _macro()  # has a row dated AS_OF, the "holiday"
    closes = _closes(macro, {"R": (0.02, -0.01)}).iloc[:-1]  # last close: the day before
    moved_macro = macro.copy()
    moved_macro.iloc[-1] = [99.0, 99.0]
    bumped = closes.copy()
    bumped.iloc[-1] = closes.iloc[-1] * 1.5
    base = factor(closes, macro, AS_OF)
    pd.testing.assert_series_equal(factor(closes, moved_macro, AS_OF), base)
    assert factor(bumped, macro, AS_OF)["R"] != pytest.approx(base["R"], rel=1e-3)


def test_a_contemporaneous_link_is_recovered_and_a_one_day_lead_lag_is_not() -> None:
    macro = _macro()
    same_day = _closes(macro, {"X": (0.02, 0.0)})
    next_day = _closes(macro, {"X": (0.02, 0.0)}, lag=1)
    assert macro_beta(same_day, macro[RATE_SERIES], AS_OF)["X"] == pytest.approx(0.02, rel=0.05)
    assert abs(macro_beta(next_day, macro[RATE_SERIES], AS_OF)["X"]) < 0.25 * 0.02


@pytest.mark.parametrize(("known", "finite"), [(MIN_PAIRS + 1, True), (MIN_PAIRS, False)])
def test_200_paired_returns_give_a_beta_and_199_give_nan(known: int, finite: bool) -> None:
    macro = _macro()
    closes = _closes(macro, {"R": (0.02, 0.0)})
    sparse = macro.copy()
    # 252 rows lie before AS_OF; keep the last `known` of them, so known - 1 returns pair up
    sparse.iloc[: len(macro) - 1 - known] = np.nan
    beta = macro_beta(closes, sparse[RATE_SERIES], AS_OF)["R"]
    assert math.isfinite(beta) is finite


def test_a_missing_macro_day_drops_that_interval_and_makes_no_zero_change() -> None:
    macro = _macro()
    closes = _closes(macro, {"R": (0.02, 0.0)})
    gap = macro.index[100]
    holed = macro.copy()
    holed.loc[gap] = np.nan
    beta = macro_beta(closes, holed[RATE_SERIES], AS_OF)
    dropped = macro_beta(closes.drop(index=gap), macro[RATE_SERIES].drop(index=gap), AS_OF)
    filled = macro_beta(closes, holed[RATE_SERIES].ffill(), AS_OF)
    pd.testing.assert_series_equal(beta, dropped)
    assert beta["R"] != pytest.approx(filled["R"], rel=1e-4)


def test_a_missing_close_drops_that_interval_and_makes_no_zero_return() -> None:
    macro = _macro()
    closes = _closes(macro, {"R": (0.02, 0.0)})
    gap = closes.index[100]
    holed = closes.copy()
    holed.loc[gap, "R"] = np.nan
    beta = macro_beta(holed, macro[RATE_SERIES], AS_OF)
    dropped = macro_beta(closes.drop(index=gap), macro[RATE_SERIES].drop(index=gap), AS_OF)
    filled = macro_beta(holed.ffill(), macro[RATE_SERIES], AS_OF)
    pd.testing.assert_series_equal(beta, dropped)
    assert beta["R"] != pytest.approx(filled["R"], rel=1e-4)


def test_closes_and_series_on_different_datetime_units_still_align() -> None:
    macro = _macro()
    closes = _closes(macro, {"R": (0.02, 0.0)})
    micro = closes.set_axis(pd.DatetimeIndex(closes.index).as_unit("us"))
    pd.testing.assert_series_equal(
        macro_beta(micro, macro[RATE_SERIES], AS_OF), macro_beta(closes, macro[RATE_SERIES], AS_OF)
    )


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("bad", [0.0, -5.0])
def test_a_non_positive_close_is_dropped_like_a_missing_one(bad: float) -> None:
    macro = _macro()
    closes = _closes(macro, {"R": (0.02, 0.0)})
    gap = closes.index[100]
    holed = closes.copy()
    holed.loc[gap, "R"] = bad
    beta = macro_beta(holed, macro[RATE_SERIES], AS_OF)
    dropped = macro_beta(closes.drop(index=gap), macro[RATE_SERIES].drop(index=gap), AS_OF)
    assert math.isfinite(beta["R"])
    pd.testing.assert_series_equal(beta, dropped)


def test_a_near_constant_series_gives_nan() -> None:
    macro = _macro()
    closes = _closes(macro, {"R": (0.02, 0.0)})
    jitter = np.where(np.arange(len(macro)) % 2 == 0, 0.0, 8.9e-16)
    near_flat = pd.Series(4.0 + jitter, index=macro.index)
    assert math.isnan(macro_beta(closes, near_flat, AS_OF)["R"])
    residue = pd.Series(0.1 + 0.2, index=macro.index)
    assert math.isnan(macro_beta(closes, residue, AS_OF)["R"])


def _walk(count: int) -> pd.Series:
    """``count`` strictly varying values on the business days ending before AS_OF."""
    index = pd.bdate_range(end=pd.Timestamp(AS_OF) - pd.Timedelta(days=1), periods=count)
    return pd.Series(np.arange(count, dtype="float64") ** 1.5, index=index)


def test_a_series_with_min_pairs_plus_one_observations_has_no_problem() -> None:
    assert macro_series_problem(_walk(MIN_PAIRS + 1), AS_OF) is None


def test_a_series_with_min_pairs_observations_is_a_problem() -> None:
    problem = macro_series_problem(_walk(MIN_PAIRS), AS_OF)
    assert problem is not None
    assert f"has {MIN_PAIRS} finite observations" in problem
    assert f"needs {MIN_PAIRS + 1}" in problem


def test_a_constant_series_is_a_problem() -> None:
    flat = pd.Series(4.0, index=_walk(MIN_PAIRS + 50).index)
    assert macro_series_problem(flat, AS_OF) == f"is constant before {AS_OF}"


def test_a_series_without_a_finite_value_is_a_problem() -> None:
    empty = pd.Series(math.nan, index=_walk(MIN_PAIRS + 50).index)
    assert macro_series_problem(empty, AS_OF) == f"has no finite value before {AS_OF}"


def test_inf_and_nan_are_not_counted_as_observations() -> None:
    series = _walk(MIN_PAIRS + 2)
    series.iloc[0] = math.nan
    assert macro_series_problem(series, AS_OF) is None
    series.iloc[1] = math.inf
    problem = macro_series_problem(series, AS_OF)
    assert problem is not None
    assert f"has {MIN_PAIRS} finite observations" in problem


def test_rows_on_or_after_as_of_are_not_counted() -> None:
    series = _walk(MIN_PAIRS)
    later = pd.Series(
        [1.0, 2.0], index=[pd.Timestamp(AS_OF), pd.Timestamp(AS_OF) + pd.Timedelta(days=1)]
    )
    problem = macro_series_problem(pd.concat([series, later]), AS_OF)
    assert problem is not None
    assert f"has {MIN_PAIRS} finite observations" in problem


def test_an_inf_in_the_series_acts_like_a_missing_day() -> None:
    macro = _macro()
    closes = _closes(macro, {"R": (0.02, 0.0)})
    with_nan = macro[RATE_SERIES].copy()
    with_nan.iloc[100] = math.nan
    with_inf = macro[RATE_SERIES].copy()
    with_inf.iloc[100] = math.inf
    expected = macro_beta(closes, with_nan, AS_OF)
    assert np.isfinite(expected["R"])
    pd.testing.assert_series_equal(macro_beta(closes, with_inf, AS_OF), expected)
    with_inf.iloc[100] = -math.inf
    pd.testing.assert_series_equal(macro_beta(closes, with_inf, AS_OF), expected)


def test_a_nullable_series_with_na_is_handled() -> None:
    macro = _macro()
    closes = _closes(macro, {"R": (0.02, 0.0)})
    plain = macro[RATE_SERIES].copy()
    plain.iloc[100] = math.nan
    nullable = plain.astype("Float64")
    assert nullable.isna().iloc[100]
    pd.testing.assert_series_equal(
        macro_beta(closes, nullable, AS_OF), macro_beta(closes, plain, AS_OF)
    )
    assert macro_series_problem(nullable, AS_OF) is None
    short = _walk(MIN_PAIRS).astype("Float64")
    short.iloc[0] = pd.NA
    problem = macro_series_problem(short, AS_OF)
    assert problem is not None
    assert f"has {MIN_PAIRS - 1} finite observations" in problem


def test_a_constant_series_gives_nan() -> None:
    macro = _macro()
    closes = _closes(macro, {"R": (0.02, 0.0)})
    flat = pd.Series(4.0, index=macro.index)
    assert math.isnan(macro_beta(closes, flat, AS_OF)["R"])
