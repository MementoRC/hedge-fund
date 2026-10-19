"""Same-side correlated-pair pre-filter: of two near-duplicate names, keep the stronger."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from hedge_fund.domain.shortlist import Side


@dataclass(frozen=True)
class CorrelationParams:
    """Same-side pairs with sample correlation above ``max_rho`` keep only one name."""

    max_rho: float = 0.85


DEFAULT_CORRELATION = CorrelationParams()


def correlation_filter(
    scores: Mapping[str, float],
    sides: Mapping[str, Side],
    corr: pd.DataFrame,
    *,
    params: CorrelationParams = DEFAULT_CORRELATION,
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Greedily keep names by ``scores`` (descending, ties by ticker).

    A name is dropped as ``correlated:<kept>`` when an already-kept same-side name has
    ``corr > max_rho`` with it (the first such name in keep order). Opposite-side pairs are
    never filtered and a NaN correlation never clashes. ``corr`` is labelled by ticker on
    both axes. Returns the kept tickers in keep order and the exclusions sorted by ticker.
    Raises ValueError on invalid params.
    """
    _validate_params(params)
    kept: list[str] = []
    excluded: dict[str, str] = {}
    for ticker in sorted(scores, key=lambda t: (-scores[t], t)):
        clash = _first_clash(ticker, kept, sides, corr, params.max_rho)
        if clash is None:
            kept.append(ticker)
        else:
            excluded[ticker] = f"correlated:{clash}"
    return tuple(kept), dict(sorted(excluded.items()))


def _first_clash(
    ticker: str, kept: list[str], sides: Mapping[str, Side], corr: pd.DataFrame, max_rho: float
) -> str | None:
    for other in kept:
        if sides[other] is sides[ticker] and corr.at[ticker, other] > max_rho:
            return other
    return None


def _validate_params(params: CorrelationParams) -> None:
    """Type check first: bool and NaN would slip past plain range comparisons."""
    value = params.max_rho
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"max_rho must be a finite number, got {value!r}")
    if not -1 < value <= 1:
        raise ValueError(f"max_rho must satisfy -1 < max_rho <= 1, got {value!r}")
