"""Factor crowding: excess residual co-movement of each factor's top and bottom names."""

from __future__ import annotations

from dataclasses import dataclass

from hedge_fund.domain.scores import CrowdingReport, ScoreResult
from hedge_fund.domain.snapshot import PriceSnapshot
from hedge_fund.engine.factors.definitions import FACTORS

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

    Call after ``score()`` with the same snapshot. Raises ValueError on mismatched inputs
    or invalid params; an empty ``result`` gives an empty report.
    """
    _validate(snapshot, result)
    _validate_params(params)
    if not result.scores:
        return CrowdingReport(as_of=result.as_of, baseline_corr=None, factors=())
    raise NotImplementedError("per-factor legs arrive in Task 3")


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
