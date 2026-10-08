"""Tests for the factor scoring pipeline."""

import numpy as np
import pandas as pd
import pytest

from hedge_fund.domain.snapshot import PriceSnapshot
from hedge_fund.engine.factors.definitions import FACTORS, FactorSpec, low_vol, momentum_12_1
from hedge_fund.engine.factors.score import score
from tests.factories import make_closes, snapshot_from

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


def _snapshot(closes: pd.DataFrame) -> PriceSnapshot:
    return snapshot_from(closes, SECTORS)


def _last_price(window: pd.DataFrame) -> pd.Series:
    return window.iloc[-1]


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
        assert set(s.raw) == {"momentum_12_1", "low_vol"}
        assert s.composite == pytest.approx(sum(s.z.values()) / 2)
        assert s.sector == SECTORS[s.ticker]
        assert s.fallback_universe_z is False


def test_raw_values_match_factor_definitions() -> None:
    closes = make_closes(TICKERS)
    result = score(_snapshot(closes))
    window = closes.iloc[-WINDOW:]
    momentum = momentum_12_1(window)
    vol = low_vol(window)
    for s in result.scores:
        assert s.raw["momentum_12_1"] == pytest.approx(momentum[s.ticker])
        assert s.raw["low_vol"] == pytest.approx(vol[s.ticker])


def test_history_before_window_is_ignored() -> None:
    closes = make_closes(TICKERS, n_days=400)
    assert score(_snapshot(closes)) == score(_snapshot(closes.iloc[-WINDOW:]))


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
    closes.iloc[:60, 0] = np.nan  # A1: insufficient_history ...
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
