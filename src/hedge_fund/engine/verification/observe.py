"""One researched name scored against the benchmark after a horizon. Pure."""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Sequence
from datetime import date

import pandas as pd

from hedge_fund.domain.research import NameResearch, Verdict
from hedge_fund.domain.shortlist import Side
from hedge_fund.domain.verification import ExclusionReason, Observation, ObservationStatus


def observe(
    name: NameResearch,
    as_of: date,
    closes: pd.Series,
    spy: pd.Series,
    calendar: Sequence[date],
    horizon: int,
    executed: bool,
) -> Observation:
    """Score ``name``'s research of ``as_of`` over ``horizon`` benchmark sessions.

    ``calendar`` is the sorted benchmark trading days; ``closes`` (the name) and ``spy`` hold
    one close per calendar day, in calendar order. The base is the last calendar day on or
    before ``as_of`` (ValueError if there is none), the end the day ``horizon`` sessions
    later. Insufficient research (verdict INSUFFICIENT or no mean) is checked first and is
    EXCLUDED with no calendar or price lookup (``base_date`` None), so it never raises; an
    end past the last calendar day is PENDING (on it, SCORED); a name close that is not
    finite and positive at the base or the end is EXCLUDED for NO_PRICE. Otherwise SCORED:
    ``aligned_excess = side * (name_return - spy_return)`` and ``hit`` is None without a
    call (``mean == 0``), else whether the signs of the mean and the aligned excess agree
    (a zero excess is a miss).
    """
    mean = name.mean
    if name.verdict is Verdict.INSUFFICIENT or mean is None:
        return _row(
            name,
            as_of,
            horizon,
            None,
            executed,
            status=ObservationStatus.EXCLUDED,
            reason=ExclusionReason.INSUFFICIENT_RESEARCH,
        )
    if len(closes) != len(calendar) or len(spy) != len(calendar):
        raise ValueError("closes and spy need one close per calendar day")
    position = bisect_right(calendar, as_of) - 1
    if position < 0:
        raise ValueError(f"no trading day on or before {as_of}")
    base = calendar[position]
    target = position + horizon
    if target >= len(calendar):
        return _row(name, as_of, horizon, base, executed, status=ObservationStatus.PENDING)
    end = calendar[target]
    first, last = _close(closes, position), _close(closes, target)
    if first is None or last is None:
        return _row(
            name,
            as_of,
            horizon,
            base,
            executed,
            end=end,
            status=ObservationStatus.EXCLUDED,
            reason=ExclusionReason.NO_PRICE,
        )
    name_return = last / first - 1.0
    spy_return = _benchmark_close(spy, target, end) / _benchmark_close(spy, position, base) - 1.0
    side = 1.0 if name.candidate.side is Side.LONG else -1.0
    aligned = side * (name_return - spy_return)
    return _row(
        name,
        as_of,
        horizon,
        base,
        executed,
        end=end,
        name_return=name_return,
        spy_return=spy_return,
        aligned_excess=aligned,
        hit=_hit(mean, aligned),
        status=ObservationStatus.SCORED,
    )


def _row(
    name: NameResearch,
    as_of: date,
    horizon: int,
    base: date | None,
    executed: bool,
    *,
    status: ObservationStatus,
    reason: ExclusionReason | None = None,
    end: date | None = None,
    name_return: float | None = None,
    spy_return: float | None = None,
    aligned_excess: float | None = None,
    hit: bool | None = None,
) -> Observation:
    return Observation(
        as_of=as_of,
        ticker=name.candidate.ticker,
        sector=name.candidate.sector,
        side=name.candidate.side,
        verdict=name.verdict,
        mean=name.mean,
        std=name.std,
        agreement=name.agreement,
        valid=name.valid,
        horizon=horizon,
        base_date=base,
        end_date=end,
        name_return=name_return,
        spy_return=spy_return,
        aligned_excess=aligned_excess,
        hit=hit,
        executed=executed,
        status=status,
        reason=reason,
    )


def _close(series: pd.Series, position: int) -> float | None:
    """The close at ``position`` if it is finite and positive, else None."""
    value = float(series.iloc[position])
    return value if math.isfinite(value) and value > 0 else None


def _benchmark_close(series: pd.Series, position: int, day: date) -> float:
    value = _close(series, position)
    if value is None:
        raise ValueError(f"no benchmark close on {day}")
    return value


def _hit(mean: float, aligned: float) -> bool | None:
    """None without a directional call; else whether the signs agree (zero excess misses)."""
    if mean == 0:
        return None
    return aligned != 0 and (mean > 0) == (aligned > 0)
