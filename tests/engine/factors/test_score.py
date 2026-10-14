"""Tests for the factor scoring pipeline."""

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from hedge_fund.domain.fundamentals import FundamentalsSnapshot
from hedge_fund.domain.snapshot import PriceSnapshot
from hedge_fund.engine.factors.definitions import (
    FACTORS,
    FactorSpec,
    high_52w_proximity,
    low_vol,
    momentum_12_1,
    reversal_1m,
)
from hedge_fund.engine.factors.fundamental import FUNDAMENTAL_FACTORS, FundamentalFactorSpec, size
from hedge_fund.engine.factors.score import score
from tests.factories import make_closes, make_fundamentals, snapshot_from

SECTORS = {
    "A1": "Tech",
    "A2": "Tech",
    "A3": "Tech",
    "A4": "Tech",
    "B1": "Energy",
    "B2": "Energy",
    "B3": "Energy",
    "B4": "Energy",
}
TICKERS = list(SECTORS)
WINDOW = 253
FACTOR_NAMES = {"momentum_12_1", "low_vol", "reversal_1m", "high_52w_proximity"}
FUNDAMENTAL_NAMES = {"earnings_yield", "gross_profitability", "revenue_growth", "size"}
TECH = ["A1", "A2", "A3", "A4"]


def _snapshot(closes: pd.DataFrame) -> PriceSnapshot:
    return snapshot_from(closes, SECTORS)


def _last_price(window: pd.DataFrame) -> pd.Series:
    return window.iloc[-1]


def _never_called(window: pd.DataFrame) -> pd.Series:
    raise AssertionError("factors must not run when nothing is scorable")


def _with_current(
    fundamentals: FundamentalsSnapshot, values: dict[str, dict[str, float]]
) -> FundamentalsSnapshot:
    """Copy of ``fundamentals`` with ``values[ticker][metric]`` written into ``current``."""
    current = fundamentals.current.copy()
    for ticker, metrics in values.items():
        for metric, value in metrics.items():
            current.loc[ticker, metric] = value
    return FundamentalsSnapshot(
        fundamentals.as_of, current, fundamentals.prior, fundamentals.periods
    )


def _flat_closes() -> pd.DataFrame:
    """Every ticker follows the same price path, so only fundamentals differ."""
    path = make_closes(["X"])["X"]
    return pd.DataFrame(dict.fromkeys(TICKERS, path))


# --- happy path -------------------------------------------------------------


def test_scores_every_clean_ticker() -> None:
    snap = _snapshot(make_closes(TICKERS))
    result = score(snap)
    assert result.as_of == snap.as_of
    assert result.excluded == {}
    assert sorted(s.ticker for s in result.scores) == sorted(TICKERS)
    composites = [s.composite for s in result.scores]
    assert composites == sorted(composites, reverse=True)
    for s in result.scores:
        assert set(s.raw) == FACTOR_NAMES
        assert set(s.z) == FACTOR_NAMES
        assert s.composite == pytest.approx(sum(s.z.values()) / 4)
        assert s.sector == SECTORS[s.ticker]
        assert s.fallback_universe_z is False


def test_raw_values_match_factor_definitions() -> None:
    closes = make_closes(TICKERS)
    result = score(_snapshot(closes))
    window = closes.iloc[-WINDOW:]
    momentum = momentum_12_1(window)
    vol = low_vol(window)
    reversal = reversal_1m(window)
    proximity = high_52w_proximity(window)
    for s in result.scores:
        assert s.raw["momentum_12_1"] == pytest.approx(momentum[s.ticker])
        assert s.raw["low_vol"] == pytest.approx(vol[s.ticker])
        assert s.raw["reversal_1m"] == pytest.approx(reversal[s.ticker])
        assert s.raw["high_52w_proximity"] == pytest.approx(proximity[s.ticker])


def test_history_before_window_is_ignored() -> None:
    closes = make_closes(TICKERS, n_days=400)
    assert score(_snapshot(closes)) == score(_snapshot(closes.iloc[-WINDOW:]))


def test_score_does_not_mutate_input() -> None:
    closes = make_closes(TICKERS)
    closes.iloc[150, 0] = np.nan
    before = closes.copy()
    score(_snapshot(closes))
    pd.testing.assert_frame_equal(closes, before)


# --- exclusions -------------------------------------------------------------


def test_short_history_excludes_everything() -> None:
    result = score(_snapshot(make_closes(TICKERS, n_days=200)))
    assert result.scores == ()
    assert result.excluded == dict.fromkeys(TICKERS, "insufficient_history")


def test_leading_nan_and_all_nan_are_insufficient_history() -> None:
    closes = make_closes(TICKERS)  # 300 rows; window starts at row 47
    closes.iloc[:60, 0] = np.nan  # A1: leading NaNs inside the window
    closes["A2"] = np.nan
    result = score(_snapshot(closes))
    assert result.excluded == {"A1": "insufficient_history", "A2": "insufficient_history"}


def test_too_many_nans_is_missing_data() -> None:
    closes = make_closes(TICKERS)
    closes.iloc[100:120, 0] = np.nan  # 20 / 253 > 5%
    result = score(_snapshot(closes))
    assert result.excluded == {"A1": "missing_data"}


def test_interior_and_trailing_nans_are_forward_filled() -> None:
    closes = make_closes(TICKERS)
    closes.iloc[[150, 151, 152], 0] = np.nan
    closes.iloc[-1, 0] = np.nan
    result = score(_snapshot(closes))
    assert result.excluded == {}
    scored = {s.ticker: s for s in result.scores}
    expected = low_vol(closes.iloc[-WINDOW:].ffill())["A1"]
    assert scored["A1"].raw["low_vol"] == pytest.approx(expected)


def test_nonpositive_price_excluded() -> None:
    closes = make_closes(TICKERS)
    closes.iloc[200, 0] = 0.0
    closes.iloc[210, 1] = -5.0
    result = score(_snapshot(closes))
    assert result.excluded == {"A1": "nonpositive_price", "A2": "nonpositive_price"}


def test_nonfinite_factor_excluded() -> None:
    closes = make_closes(TICKERS)
    closes.iloc[200, 0] = np.inf
    result = score(_snapshot(closes))
    assert result.excluded == {"A1": "nonfinite_factor"}
    assert "A1" not in {s.ticker for s in result.scores}


def test_all_nonfinite_returns_empty_scores() -> None:
    closes = make_closes(TICKERS)
    closes.iloc[200] = np.inf
    result = score(_snapshot(closes))
    assert result.scores == ()
    assert result.excluded == dict.fromkeys(TICKERS, "nonfinite_factor")


def test_first_failing_reason_wins() -> None:
    closes = make_closes(TICKERS)
    # A1: insufficient_history (13/253 NaN in window also exceeds missing_data) ...
    closes.iloc[:60, 0] = np.nan
    closes.iloc[200, 0] = 0.0  # ... beats nonpositive_price
    closes.iloc[100:120, 1] = np.nan  # A2: missing_data ...
    closes.iloc[200, 1] = -1.0  # ... beats nonpositive_price
    closes.iloc[150, 2] = np.inf  # A3: nonpositive_price ...
    closes.iloc[160, 2] = 0.0  # ... beats nonfinite_factor
    result = score(_snapshot(closes))
    assert result.excluded == {
        "A1": "insufficient_history",
        "A2": "missing_data",
        "A3": "nonpositive_price",
    }


# --- contract errors --------------------------------------------------------


def test_duplicate_factor_names_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate factor"):
        score(_snapshot(make_closes(TICKERS)), factors=(FACTORS[0], FACTORS[0]))


def test_empty_factor_list_rejected() -> None:
    with pytest.raises(ValueError, match="at least one factor"):
        score(_snapshot(make_closes(TICKERS)), factors=())


def test_snapshot_with_future_rows_rejected() -> None:
    closes = make_closes(TICKERS)
    with pytest.raises(ValueError, match="after as_of"):
        PriceSnapshot(as_of=closes.index[-2].date(), closes=closes, sectors=SECTORS)


# --- determinism and invariants --------------------------------------------


def test_ties_broken_by_ticker() -> None:
    closes = make_closes(TICKERS)
    closes["A2"] = closes["A1"]
    closes = closes[closes.columns[::-1]]
    order = [s.ticker for s in score(_snapshot(closes)).scores]
    assert order.index("A2") == order.index("A1") + 1


def test_prices_after_as_of_cannot_affect_scores() -> None:
    full = make_closes(TICKERS, n_days=400)
    cutoff = full.index[300]
    perturbed = full.copy()
    perturbed.loc[perturbed.index > cutoff] *= 3.0
    base = score(_snapshot(full.loc[:cutoff]))
    moved = score(_snapshot(perturbed.loc[:cutoff]))
    assert base == moved


def test_column_order_does_not_matter() -> None:
    closes = make_closes(TICKERS)
    base = score(_snapshot(closes))
    shuffled = score(_snapshot(closes[list(reversed(TICKERS))]))
    assert [s.ticker for s in shuffled.scores] == [s.ticker for s in base.scores]
    for a, b in zip(base.scores, shuffled.scores, strict=True):
        assert b.composite == pytest.approx(a.composite)
        assert b.z == pytest.approx(a.z)


def test_negative_sign_inverts_ranking_but_keeps_raw() -> None:
    closes = make_closes(TICKERS)
    spec = FactorSpec("last_price", _last_price, sign=-1, min_history=WINDOW)
    result = score(_snapshot(closes), factors=(spec,))
    last = closes.iloc[-1]
    for s in result.scores:
        assert s.raw["last_price"] == pytest.approx(last[s.ticker])
    tech = [s for s in result.scores if s.sector == "Tech"]
    cheapest = min(tech, key=lambda s: s.raw["last_price"])
    assert cheapest.z["last_price"] == max(s.z["last_price"] for s in tech)


def test_small_sector_flags_fallback() -> None:
    sectors = {**SECTORS, "C1": "Utilities", "C2": "Utilities"}
    closes = make_closes([*TICKERS, "C1", "C2"])
    result = score(snapshot_from(closes, sectors))
    flags = {s.ticker: s.fallback_universe_z for s in result.scores}
    assert flags.pop("C1") is True
    assert flags.pop("C2") is True
    assert not any(flags.values())


# --- fundamentals: contract -------------------------------------------------


def test_fundamentals_as_of_mismatch_rejected() -> None:
    snap = _snapshot(make_closes(TICKERS).iloc[:-1])
    with pytest.raises(ValueError, match="as_of"):
        score(snap, make_fundamentals(TICKERS))


def test_duplicate_name_across_factor_lists_rejected() -> None:
    snap = _snapshot(make_closes(TICKERS))
    clash = FundamentalFactorSpec("low_vol", size, sign=-1)
    with pytest.raises(ValueError, match="duplicate factor"):
        score(snap, make_fundamentals(TICKERS), fundamental_factors=(clash,))


def test_price_factors_required_with_fundamentals() -> None:
    snap = _snapshot(make_closes(TICKERS))
    with pytest.raises(ValueError, match="at least one factor"):
        score(snap, make_fundamentals(TICKERS), factors=())


def test_fundamental_factors_ignored_without_fundamentals() -> None:
    snap = _snapshot(make_closes(TICKERS))
    clash = FundamentalFactorSpec("low_vol", size, sign=-1)
    assert score(snap, fundamental_factors=(clash,)) == score(snap)


def test_empty_fundamental_factors_is_price_only_scoring() -> None:
    snap = _snapshot(make_closes(TICKERS))
    assert score(
        snap, make_fundamentals(TICKERS), fundamental_factors=(), quality_gates=None
    ) == score(snap)


# --- fundamentals: exclusions -----------------------------------------------
# fundamental_factors=() isolates the exclusion rule from the factor join.


def test_missing_fundamentals_excluded() -> None:
    fundamentals = make_fundamentals([t for t in TICKERS if t != "A1"])
    result = score(
        _snapshot(make_closes(TICKERS)), fundamentals, fundamental_factors=(), quality_gates=None
    )
    assert result.excluded == {"A1": "missing_fundamentals"}
    assert "A1" not in {s.ticker for s in result.scores}


def test_price_exclusion_beats_missing_fundamentals() -> None:
    closes = make_closes(TICKERS)
    closes.iloc[200, 0] = 0.0  # A1: nonpositive_price and no fundamentals row
    fundamentals = make_fundamentals([t for t in TICKERS if t not in {"A1", "A2"}])
    result = score(_snapshot(closes), fundamentals, fundamental_factors=(), quality_gates=None)
    assert result.excluded == {"A1": "nonpositive_price", "A2": "missing_fundamentals"}


def test_all_missing_fundamentals_returns_before_computing_factors() -> None:
    spec = FactorSpec("never", _never_called, sign=1, min_history=WINDOW)
    result = score(_snapshot(make_closes(TICKERS)), make_fundamentals(["ZZZ"]), factors=(spec,))
    assert result.scores == ()
    assert result.excluded == dict.fromkeys(TICKERS, "missing_fundamentals")


# --- fundamentals: scoring --------------------------------------------------


def test_fundamentals_add_four_scored_factors() -> None:
    result = score(_snapshot(make_closes(TICKERS)), make_fundamentals(TICKERS), quality_gates=None)
    expected_order = [spec.name for spec in FACTORS] + [spec.name for spec in FUNDAMENTAL_FACTORS]
    assert result.excluded == {}
    assert len(result.scores) == len(TICKERS)
    for s in result.scores:
        assert list(s.raw) == expected_order
        assert set(s.z) == FACTOR_NAMES | FUNDAMENTAL_NAMES
        assert s.composite == pytest.approx(sum(s.z.values()) / 8)


def test_fundamental_raw_values_follow_ticker_labels() -> None:
    closes = make_closes(TICKERS)[list(reversed(TICKERS))]
    fundamentals = make_fundamentals(TICKERS)
    current, prior = fundamentals.current, fundamentals.prior
    cap = current["shares_outstanding"] * closes.iloc[-1]
    result = score(_snapshot(closes), fundamentals, quality_gates=None)
    for s in result.scores:
        t = s.ticker
        profitability = current.at[t, "gross_profit"] / current.at[t, "total_assets"]
        growth = current.at[t, "revenue"] / prior.at[t, "revenue"] - 1.0
        assert s.raw["earnings_yield"] == pytest.approx(current.at[t, "net_income"] / cap[t])
        assert s.raw["gross_profitability"] == pytest.approx(profitability)
        assert s.raw["revenue_growth"] == pytest.approx(growth)
        assert s.raw["size"] == pytest.approx(np.log(cap[t]))


def test_fundamentals_only_tickers_ignored() -> None:
    result = score(
        _snapshot(make_closes(TICKERS)), make_fundamentals([*TICKERS, "ZZZ"]), quality_gates=None
    )
    assert result.excluded == {}
    assert sorted(s.ticker for s in result.scores) == sorted(TICKERS)
    assert all(set(s.raw) >= FUNDAMENTAL_NAMES for s in result.scores)


def test_nan_gross_profit_is_nonfinite_factor() -> None:
    fundamentals = _with_current(make_fundamentals(TICKERS), {"A1": {"gross_profit": np.nan}})
    result = score(_snapshot(make_closes(TICKERS)), fundamentals, quality_gates=None)
    assert result.excluded == {"A1": "nonfinite_factor"}
    assert "A1" not in {s.ticker for s in result.scores}


def test_nan_prior_metric_other_than_revenue_is_not_excluded() -> None:
    fundamentals = make_fundamentals(TICKERS)
    prior = fundamentals.prior.copy()
    prior.loc["A1", "net_income"] = np.nan
    patched = FundamentalsSnapshot(
        fundamentals.as_of, fundamentals.current, prior, fundamentals.periods
    )
    result = score(_snapshot(make_closes(TICKERS)), patched, quality_gates=None)
    assert "A1" in {s.ticker for s in result.scores}
    assert "A1" not in result.excluded


def test_lower_earnings_yield_gets_lower_z() -> None:
    net_income = {"A1": 1.0, "A2": 2.0, "A3": 3.0, "A4": 4.0}
    values = {
        t: {"shares_outstanding": 10.0, "net_income": net_income.get(t, 5.0)} for t in TICKERS
    }
    fundamentals = _with_current(make_fundamentals(TICKERS), values)
    result = score(_snapshot(_flat_closes()), fundamentals, quality_gates=None)
    z = {s.ticker: s.z["earnings_yield"] for s in result.scores}
    assert all(z[low] < z[high] for low, high in pairwise(TECH))


def test_larger_market_cap_gets_lower_size_z() -> None:
    shares = {"A1": 10.0, "A2": 20.0, "A3": 30.0, "A4": 40.0}
    values = {
        t: {"shares_outstanding": shares.get(t, 25.0), "net_income": shares.get(t, 25.0)}
        for t in TICKERS
    }  # net income scales with shares, so earnings yield is equal across tickers
    fundamentals = _with_current(make_fundamentals(TICKERS), values)
    result = score(_snapshot(_flat_closes()), fundamentals, quality_gates=None)
    by_ticker = {s.ticker: s for s in result.scores}
    yields = [by_ticker[t].raw["earnings_yield"] for t in TECH]
    assert yields == pytest.approx([yields[0]] * len(TECH))
    z = {t: s.z["size"] for t, s in by_ticker.items()}
    assert all(z[small] > z[large] for small, large in pairwise(TECH))
