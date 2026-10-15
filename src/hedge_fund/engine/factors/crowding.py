"""Factor crowding: excess residual co-movement of each factor's top and bottom names."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from hedge_fund.domain.scores import CrowdingReport, FactorCrowding, FactorScore, ScoreResult
from hedge_fund.domain.snapshot import PriceSnapshot
from hedge_fund.engine.factors.definitions import FACTORS
from hedge_fund.engine.factors.normalize import MIN_SECTOR_SIZE

WINDOW_ROWS = max(spec.min_history for spec in FACTORS)


@dataclass(frozen=True)
class CrowdingParams:
    """Leg size, measurement floor, flag threshold and return window for ``crowding``.

    ``window_rows`` matches ``score()`` for the default factor registry; when scoring with
    custom factors pass ``max(f.min_history for f in factors)``.
    """

    quantile: float = 0.2
    min_leg: int = 5
    threshold: float = 0.10
    window_rows: int = WINDOW_ROWS


DEFAULT_CROWDING = CrowdingParams()


def crowding(
    snapshot: PriceSnapshot, result: ScoreResult, *, params: CrowdingParams = DEFAULT_CROWDING
) -> CrowdingReport:
    """Excess residual correlation of each factor's long and short legs (diagnostic only).

    Call after ``score()`` with the same snapshot. Returns are daily log returns of the last
    ``params.window_rows`` forward-filled closes, minus the equal-weight sector mean (the
    universe mean for sectors under MIN_SECTOR_SIZE scored names). Raises ValueError on
    mismatched inputs or invalid params; an empty ``result`` gives an empty report.
    """
    _validate(snapshot, result)
    _validate_params(params)
    if not result.scores:
        return CrowdingReport(as_of=result.as_of, baseline_corr=None, factors=())
    sectors = {s.ticker: s.sector for s in result.scores}
    window = snapshot.closes[list(sectors)].iloc[-params.window_rows :].ffill()
    returns = np.log(window).diff().iloc[1:]
    corr = _residuals(returns, sectors).corr()
    baseline = _mean_pair_corr(corr, list(sectors))
    factors = tuple(
        _factor_crowding(name, result.scores, corr, baseline, params) for name in result.scores[0].z
    )
    return CrowdingReport(as_of=result.as_of, baseline_corr=baseline, factors=factors)


def _validate(snapshot: PriceSnapshot, result: ScoreResult) -> None:
    """Checks 1-3: same as_of, one z key set, every scored ticker has closes."""
    if snapshot.as_of != result.as_of:
        raise ValueError(f"snapshot as_of {snapshot.as_of} != result as_of {result.as_of}")
    mismatched = [s.ticker for s in result.scores if set(s.z) != set(result.scores[0].z)]
    if mismatched:
        raise ValueError(f"z keys differ from {result.scores[0].ticker}: {mismatched}")
    missing = [s.ticker for s in result.scores if s.ticker not in snapshot.closes.columns]
    if missing:
        raise ValueError(f"scored tickers missing from closes: {missing}")


def _validate_params(params: CrowdingParams) -> None:
    """Checks 4-7, written as negated range checks so NaN is rejected."""
    if not 0 < params.quantile <= 0.5:
        raise ValueError(f"quantile must be in (0, 0.5], got {params.quantile}")
    if params.min_leg < 2:
        raise ValueError(f"min_leg must be >= 2, got {params.min_leg}")
    if not 0 < params.threshold <= 1:
        raise ValueError(f"threshold must be in (0, 1], got {params.threshold}")
    if params.window_rows < 2:
        raise ValueError(f"window_rows must be >= 2, got {params.window_rows}")


def _residuals(returns: pd.DataFrame, sectors: Mapping[str, str]) -> pd.DataFrame:
    """Returns minus the equal-weight sector mean; universe mean for small sectors."""
    members: dict[str, list[str]] = {}
    for ticker in returns.columns:
        members.setdefault(sectors[ticker], []).append(ticker)
    universe = returns.mean(axis=1)
    sector_mean = {
        sector: returns[peers].mean(axis=1) if len(peers) >= MIN_SECTOR_SIZE else universe
        for sector, peers in members.items()
    }
    benchmark = pd.DataFrame({t: sector_mean[sectors[t]] for t in returns.columns})
    return returns - benchmark


def _mean_pair_corr(corr: pd.DataFrame, tickers: Sequence[str]) -> float | None:
    """Mean of the finite off-diagonal correlations among ``tickers``; None if none."""
    values = corr.loc[list(tickers), list(tickers)].to_numpy()
    pairs = values[np.triu_indices(len(tickers), k=1)]
    finite = pairs[np.isfinite(pairs)]
    if finite.size == 0:
        return None
    return float(finite.mean())


def _leg(scores: Sequence[FactorScore], name: str, m: int, *, sign: int) -> list[str]:
    """The ``m`` tickers with the highest ``sign * z[name]``; ties by ticker ascending."""
    ranked = sorted(scores, key=lambda score: (-sign * score.z[name], score.ticker))
    return [score.ticker for score in ranked[:m]]


def _excess(leg_mean: float | None, baseline: float | None) -> float | None:
    """``leg_mean - baseline``, or None when either is None."""
    if leg_mean is None or baseline is None:
        return None
    return leg_mean - baseline


def _factor_crowding(
    name: str,
    scores: Sequence[FactorScore],
    corr: pd.DataFrame,
    baseline: float | None,
    params: CrowdingParams,
) -> FactorCrowding:
    """Long (top ``m`` by z) and short (bottom ``m``) leg excess for one factor.

    Legs with fewer than ``params.min_leg`` names are not measured (None excess).
    """
    m = int(len(scores) * params.quantile + 1e-9)  # 100 * 0.29 == 28.999999999999996
    if m < params.min_leg:
        return FactorCrowding(name, None, None, m, m, crowded=False)
    long_excess = _excess(_mean_pair_corr(corr, _leg(scores, name, m, sign=1)), baseline)
    short_excess = _excess(_mean_pair_corr(corr, _leg(scores, name, m, sign=-1)), baseline)
    crowded = any(e is not None and e >= params.threshold for e in (long_excess, short_excess))
    return FactorCrowding(name, long_excess, short_excess, m, m, crowded)
