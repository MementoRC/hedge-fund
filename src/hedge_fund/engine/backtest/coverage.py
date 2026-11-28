"""How much of the index membership history has price data (pure).

Survivorship bias shows as member-days with no stored close. ``measure_coverage`` counts
them per membership span, so a run can state the fraction it actually priced.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from hedge_fund.domain.backtest import FlaggedSpan, MemberSpan


@dataclass(frozen=True)
class SpanCoverage:
    """One span's counts over the window.

    ``window_days`` are the calendar days in the window the span holds and no flagged span
    of the ticker covers; ``flagged_days`` the ones a flagged span covers; ``priced_days``
    the window days with a stored close. ``coverage`` is priced/window days, None when
    every day is flagged.
    """

    ticker: str
    start: date
    end: date | None
    window_days: int
    flagged_days: int
    priced_days: int
    first_priced: date | None

    @property
    def coverage(self) -> float | None:
        return self.priced_days / self.window_days if self.window_days else None


@dataclass(frozen=True)
class CoverageReport:
    """Coverage of the membership history over ``start..end`` inclusive.

    ``spans`` are sorted by ticker, then start; a span with no window day and no flagged day
    is left out. A span counts as zero, partial or full by its priced share of ``window_days``;
    an all-flagged span counts in none. ``coverage`` is ``priced_member_days / member_days``
    (flagged days excluded), None when there are no member-days.
    """

    start: date
    end: date
    spans: tuple[SpanCoverage, ...]
    member_days: int
    priced_member_days: int
    flagged_member_days: int
    zero_spans: int
    partial_spans: int
    full_spans: int

    @property
    def coverage(self) -> float | None:
        return self.priced_member_days / self.member_days if self.member_days else None


def measure_coverage(
    spans: Iterable[MemberSpan],
    calendar: Sequence[date],
    priced: Mapping[str, Collection[date]],
    flagged: Iterable[FlaggedSpan],
    start: date,
    end: date,
) -> CoverageReport:
    """The report of ``spans`` over the calendar days in ``start..end`` inclusive."""
    days = [day for day in calendar if start <= day <= end]
    flagged_by_ticker: dict[str, list[MemberSpan]] = {}
    for item in flagged:
        flagged_by_ticker.setdefault(item.span.ticker, []).append(item.span)
    candidates = (
        _row(span, days, priced.get(span.ticker, ()), flagged_by_ticker.get(span.ticker, ()))
        for span in sorted(spans, key=lambda s: (s.ticker, s.start))
    )
    rows = [row for row in candidates if row is not None]
    live = [row for row in rows if row.window_days]
    return CoverageReport(
        start=start,
        end=end,
        spans=tuple(rows),
        member_days=sum(row.window_days for row in rows),
        priced_member_days=sum(row.priced_days for row in rows),
        flagged_member_days=sum(row.flagged_days for row in rows),
        zero_spans=sum(row.priced_days == 0 for row in live),
        partial_spans=sum(0 < row.priced_days < row.window_days for row in live),
        full_spans=sum(row.priced_days == row.window_days for row in live),
    )


def _row(
    span: MemberSpan,
    days: Sequence[date],
    priced: Collection[date],
    flagged: Sequence[MemberSpan],
) -> SpanCoverage | None:
    held = [day for day in days if span.contains(day)]
    flagged_days = [day for day in held if any(f.contains(day) for f in flagged)]
    flagged_set = set(flagged_days)
    window = [day for day in held if day not in flagged_set]
    if not window and not flagged_days:
        return None
    hits = [day for day in window if day in priced]
    return SpanCoverage(
        ticker=span.ticker,
        start=span.start,
        end=span.end,
        window_days=len(window),
        flagged_days=len(flagged_days),
        priced_days=len(hits),
        first_priced=min(hits) if hits else None,
    )
