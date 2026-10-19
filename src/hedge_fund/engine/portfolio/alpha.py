"""Verdict gate, conviction score and expected alpha for researched names."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from hedge_fund.domain.research import NameResearch, Verdict
from hedge_fund.domain.shortlist import Side


@dataclass(frozen=True)
class AlphaParams:
    """Information coefficient turning volatility times conviction into expected alpha."""

    ic: float = 0.05


DEFAULT_ALPHA = AlphaParams()


def gate(names: Sequence[NameResearch]) -> tuple[tuple[NameResearch, ...], dict[str, str]]:
    """CONFIRM names in input order, and every other name as ``verdict:<value>``."""
    eligible = tuple(name for name in names if name.verdict is Verdict.CONFIRM)
    excluded = {
        name.candidate.ticker: f"verdict:{name.verdict.value}"
        for name in names
        if name.verdict is not Verdict.CONFIRM
    }
    return eligible, excluded


def score(name: NameResearch) -> float:
    """``(mean / 2) / (1 + std)`` from the side-aligned stats; > 0 for any CONFIRM name.

    Raises ValueError unless ``name`` is CONFIRM with ``mean`` and ``std`` set.
    """
    if name.verdict is not Verdict.CONFIRM or name.mean is None or name.std is None:
        raise ValueError(f"score needs a CONFIRM name with stats, got {name.candidate.ticker!r}")
    return (name.mean / 2) / (1 + name.std)


def alphas(
    names: Sequence[NameResearch],
    vols: Mapping[str, float],
    *,
    params: AlphaParams = DEFAULT_ALPHA,
) -> dict[str, float]:
    """``side * ic * vol * score`` per ticker (side +1 long, -1 short).

    ``vols`` are annualized volatilities from the risk model, keyed by ticker. Raises
    ValueError on invalid params or a name ``score`` rejects.
    """
    _validate_params(params)
    out: dict[str, float] = {}
    for name in names:
        ticker = name.candidate.ticker
        side = 1.0 if name.candidate.side is Side.LONG else -1.0
        out[ticker] = side * params.ic * vols[ticker] * score(name)
    return out


def _validate_params(params: AlphaParams) -> None:
    """Type check first: bool and NaN would slip past a plain ``> 0`` comparison."""
    ic = params.ic
    if isinstance(ic, bool) or not isinstance(ic, int | float) or not math.isfinite(ic):
        raise ValueError(f"ic must be a finite number, got {ic!r}")
    if not ic > 0:
        raise ValueError(f"ic must be > 0, got {ic!r}")
