"""Sector-neutral factor scoring pipeline."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from hedge_fund.domain.fundamentals import FundamentalsSnapshot
from hedge_fund.domain.scores import FactorScore, ScoreResult
from hedge_fund.domain.snapshot import PriceSnapshot
from hedge_fund.engine.factors.definitions import FACTORS, FactorSpec
from hedge_fund.engine.factors.fundamental import (
    FUNDAMENTAL_FACTORS,
    FundamentalFactorSpec,
    FundamentalInputs,
)
from hedge_fund.engine.factors.normalize import sector_zscore, winsorize
from hedge_fund.engine.factors.quality import DEFAULT_GATES, QualityGates, altman_z, piotroski

MAX_MISSING_FRACTION = 0.05


def score(
    snapshot: PriceSnapshot,
    fundamentals: FundamentalsSnapshot | None = None,
    factors: Sequence[FactorSpec] = FACTORS,
    fundamental_factors: Sequence[FundamentalFactorSpec] = FUNDAMENTAL_FACTORS,
    quality_gates: QualityGates | None = DEFAULT_GATES,
) -> ScoreResult:
    """Score every ticker in ``snapshot`` on price factors.

    When ``fundamentals`` is given, also score on ``fundamental_factors`` (ignored when None)
    and, unless ``quality_gates`` is None, exclude tickers failing the Piotroski / Altman
    gates (``nonfinite_quality``, ``low_piotroski``, ``altman_distress``) before z-scoring.
    Unscorable tickers land in ``excluded``, including ``missing_fundamentals`` for tickers
    with no row in the fundamentals snapshot. Raises ValueError on empty factors, an
    ``as_of`` mismatch, or duplicate factor names.
    """
    active: Sequence[FundamentalFactorSpec] = ()
    if fundamentals is not None:
        active = fundamental_factors
    signs = _signs(snapshot, fundamentals, factors, active)
    names = list(signs)

    window_size = max(spec.min_history for spec in factors)
    window = snapshot.closes.iloc[-window_size:]

    excluded: dict[str, str] = {}
    for ticker in window.columns:
        reason = _pre_exclusion_reason(window[ticker], window_size)
        reason = reason or _fundamentals_reason(ticker, fundamentals)
        if reason is not None:
            excluded[ticker] = reason

    scorable = [t for t in window.columns if t not in excluded]
    if not scorable:
        return _result(snapshot, (), excluded)

    filled = window[scorable].ffill()
    inputs = None if fundamentals is None else _fundamental_inputs(fundamentals, filled)
    raw = _raw_frame(filled, inputs, factors, active)
    finite = np.isfinite(raw).all(axis=1)
    for ticker in raw.index[~finite]:
        excluded[ticker] = "nonfinite_factor"
    raw = raw[finite]
    if raw.empty:
        return _result(snapshot, (), excluded)

    f_score: pd.Series | None = None
    z_score: pd.Series | None = None
    if inputs is not None and quality_gates is not None:
        f_score, z_score, gate_excluded = _apply_gates(inputs, raw.index, quality_gates)
        excluded.update(gate_excluded)
        raw = raw.drop(index=list(gate_excluded))
        if raw.empty:
            return _result(snapshot, (), excluded)

    sectors = pd.Series({t: snapshot.sectors[t] for t in raw.index})
    z = pd.DataFrame(index=raw.index)
    fallback = pd.Series(False, index=raw.index)
    for name, sign in signs.items():
        z_factor, fallback_factor = sector_zscore(winsorize(raw[name] * sign), sectors)
        z[name] = z_factor
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
            piotroski=None if f_score is None else int(f_score[t]),
            altman_z=None if z_score is None else float(z_score[t]),
        )
        for t in raw.index
    ]
    scores.sort(key=lambda s: (-s.composite, s.ticker))
    return _result(snapshot, tuple(scores), excluded)


def _signs(
    snapshot: PriceSnapshot,
    fundamentals: FundamentalsSnapshot | None,
    factors: Sequence[FactorSpec],
    active: Sequence[FundamentalFactorSpec],
) -> dict[str, int]:
    """Validate the inputs; return the sign of every scored factor, price factors first."""
    if not factors:
        raise ValueError("score needs at least one factor")
    if fundamentals is not None and fundamentals.as_of != snapshot.as_of:
        raise ValueError(
            f"fundamentals as_of {fundamentals.as_of} != snapshot as_of {snapshot.as_of}"
        )
    names = [spec.name for spec in factors] + [spec.name for spec in active]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate factor names: {names}")
    return {spec.name: spec.sign for spec in factors} | {spec.name: spec.sign for spec in active}


def _fundamental_inputs(
    fundamentals: FundamentalsSnapshot, filled: pd.DataFrame
) -> FundamentalInputs:
    """Fundamentals reindexed to the scorable tickers, plus the last forward-filled close."""
    return FundamentalInputs(
        current=fundamentals.current.reindex(filled.columns),
        prior=fundamentals.prior.reindex(filled.columns),
        close=filled.iloc[-1],
    )


def _raw_frame(
    filled: pd.DataFrame,
    inputs: FundamentalInputs | None,
    factors: Sequence[FactorSpec],
    active: Sequence[FundamentalFactorSpec],
) -> pd.DataFrame:
    """Raw value per scorable ticker (rows) and factor (columns, price factors first)."""
    price_raw = pd.DataFrame({spec.name: spec.fn(filled) for spec in factors})
    if inputs is None:
        return price_raw
    fundamental_raw = pd.DataFrame(
        {spec.name: spec.fn(inputs) for spec in active}, index=price_raw.index
    )
    return pd.concat([price_raw, fundamental_raw], axis=1)


def _apply_gates(
    inputs: FundamentalInputs, survivors: pd.Index, gates: QualityGates
) -> tuple[pd.Series, pd.Series, dict[str, str]]:
    """Piotroski and Altman scores for ``survivors``, plus the gate exclusion reasons."""
    restricted = FundamentalInputs(
        current=inputs.current.loc[survivors],
        prior=inputs.prior.loc[survivors],
        close=inputs.close.loc[survivors],
    )
    f_score = piotroski(restricted)
    z_score = altman_z(restricted)
    excluded: dict[str, str] = {}
    for ticker in survivors:
        reason = _gate_reason(int(f_score[ticker]), float(z_score[ticker]), gates)
        if reason is not None:
            excluded[ticker] = reason
    return f_score, z_score, excluded


def _gate_reason(f_score: int, z_score: float, gates: QualityGates) -> str | None:
    """First failing quality gate, or None."""
    if not math.isfinite(z_score):
        return "nonfinite_quality"
    if f_score < gates.min_piotroski:
        return "low_piotroski"
    if z_score < gates.min_altman_z:
        return "altman_distress"
    return None


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


def _fundamentals_reason(ticker: str, fundamentals: FundamentalsSnapshot | None) -> str | None:
    """``missing_fundamentals`` when fundamentals are given but have no row for ``ticker``."""
    if fundamentals is None or ticker in fundamentals.current.index:
        return None
    return "missing_fundamentals"


def _result(
    snapshot: PriceSnapshot, scores: tuple[FactorScore, ...], excluded: dict[str, str]
) -> ScoreResult:
    return ScoreResult(as_of=snapshot.as_of, scores=scores, excluded=dict(sorted(excluded.items())))
