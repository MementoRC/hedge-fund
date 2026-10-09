"""Sector-neutral factor scoring pipeline."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from hedge_fund.domain.scores import FactorScore, ScoreResult
from hedge_fund.domain.snapshot import PriceSnapshot
from hedge_fund.engine.factors.definitions import FACTORS, FactorSpec
from hedge_fund.engine.factors.normalize import sector_zscore, winsorize

MAX_MISSING_FRACTION = 0.05


def score(snapshot: PriceSnapshot, factors: Sequence[FactorSpec] = FACTORS) -> ScoreResult:
    """Score every ticker in ``snapshot``; unscorable tickers land in ``excluded``."""
    if not factors:
        raise ValueError("score needs at least one factor")
    names = [spec.name for spec in factors]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate factor names: {names}")

    window_size = max(spec.min_history for spec in factors)
    window = snapshot.closes.iloc[-window_size:]

    excluded: dict[str, str] = {}
    for ticker in window.columns:
        reason = _pre_exclusion_reason(window[ticker], window_size)
        if reason is not None:
            excluded[ticker] = reason

    scorable = [t for t in window.columns if t not in excluded]
    if not scorable:
        return _result(snapshot, (), excluded)

    filled = window[scorable].ffill()
    raw = pd.DataFrame({spec.name: spec.fn(filled) for spec in factors})
    finite = np.isfinite(raw).all(axis=1)
    for ticker in raw.index[~finite]:
        excluded[ticker] = "nonfinite_factor"
    raw = raw[finite]
    if raw.empty:
        return _result(snapshot, (), excluded)

    sectors = pd.Series({t: snapshot.sectors[t] for t in raw.index})
    z = pd.DataFrame(index=raw.index)
    fallback = pd.Series(False, index=raw.index)
    for spec in factors:
        z_factor, fallback_factor = sector_zscore(winsorize(raw[spec.name] * spec.sign), sectors)
        z[spec.name] = z_factor
        fallback = fallback | fallback_factor
    composite = z.mean(axis=1)

    scores = [
        FactorScore(
            ticker=t,
            sector=snapshot.sectors[t],
            raw={n: float(raw.at[t, n]) for n in names},
            z={n: float(z.at[t, n]) for n in names},
            composite=float(composite[t]),
            fallback_universe_z=bool(fallback[t]),
        )
        for t in raw.index
    ]
    scores.sort(key=lambda s: (-s.composite, s.ticker))
    return _result(snapshot, tuple(scores), excluded)


def _pre_exclusion_reason(prices: pd.Series, window_size: int) -> str | None:
    """First failing data-quality rule on the unfilled window, or None."""
    first_valid = prices.first_valid_index()
    if len(prices) < window_size or first_valid is None or first_valid > prices.index[0]:
        return "insufficient_history"
    if prices.isna().mean() > MAX_MISSING_FRACTION:
        return "missing_data"
    if (prices.dropna() <= 0).any():
        return "nonpositive_price"
    return None


def _result(
    snapshot: PriceSnapshot, scores: tuple[FactorScore, ...], excluded: dict[str, str]
) -> ScoreResult:
    return ScoreResult(as_of=snapshot.as_of, scores=scores, excluded=dict(sorted(excluded.items())))
