"""Sector-balanced long/short candidate shortlist from a ScoreResult."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from hedge_fund.domain.scores import FactorScore, ScoreResult
from hedge_fund.domain.shortlist import TRIMMED_CAP, Candidate, Shortlist, Side


@dataclass(frozen=True)
class ShortlistParams:
    """Names per side per sector, and the hard cap on the whole shortlist."""

    per_sector: int = 3
    max_names: int = 40


DEFAULT_SHORTLIST = ShortlistParams()


def shortlist(result: ScoreResult, *, params: ShortlistParams = DEFAULT_SHORTLIST) -> Shortlist:
    """Pick up to ``per_sector`` longs and shorts per sector, capped at ``max_names``.

    Longs need ``composite > 0`` and shorts ``composite < 0`` (0 and NaN are on neither
    side). Candidates are taken round-robin by depth across sectors and sides, strongest
    ``abs(composite)`` first within a depth (ties by ticker), until ``max_names``; the
    qualified names left over are reported in ``trimmed`` as ``"cap"``. Raises ValueError
    on invalid params; an empty ``result`` gives an empty shortlist.
    """
    _validate_params(params)
    by_sector: dict[str, list[FactorScore]] = defaultdict(list)
    for score in result.scores:
        by_sector[score.sector].append(score)
    queues = [
        queue
        for sector in sorted(by_sector)
        for queue in _queues(by_sector[sector], params.per_sector)
    ]
    selected: list[Candidate] = []
    trimmed: dict[str, str] = {}
    for depth in range(params.per_sector):
        batch = [queue[depth] for queue in queues if depth < len(queue)]
        for candidate in sorted(batch, key=lambda c: (-abs(c.composite), c.ticker)):
            if len(selected) < params.max_names:
                selected.append(candidate)
            else:
                trimmed[candidate.ticker] = TRIMMED_CAP
    return Shortlist(
        as_of=result.as_of, candidates=tuple(selected), trimmed=dict(sorted(trimmed.items()))
    )


def _queues(scores: list[FactorScore], per_sector: int) -> tuple[list[Candidate], list[Candidate]]:
    """One sector's long and short queues, strongest first, cut to ``per_sector``."""
    longs = sorted((s for s in scores if s.composite > 0), key=lambda s: (-s.composite, s.ticker))
    shorts = sorted((s for s in scores if s.composite < 0), key=lambda s: (s.composite, s.ticker))
    return _ranked(longs, Side.LONG, per_sector), _ranked(shorts, Side.SHORT, per_sector)


def _ranked(scores: list[FactorScore], side: Side, per_sector: int) -> list[Candidate]:
    return [
        Candidate(ticker=s.ticker, sector=s.sector, side=side, depth=depth, composite=s.composite)
        for depth, s in enumerate(scores[:per_sector], start=1)
    ]


def _validate_params(params: ShortlistParams) -> None:
    """Type check first: bool, float and NaN would slip past a plain ``< 1`` comparison."""
    for name in ("per_sector", "max_names"):
        value = getattr(params, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be >= 1, got {value!r}")
