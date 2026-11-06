"""Point-in-time index membership from dated spans (pure).

A span is ``[start, end)`` in dataset snapshot dates, each the first snapshot on or after
the effective date (see domain MemberSpan). The text format is
'# ...' provenance lines, then a 'ticker,start,end' CSV with an empty ``end`` for a
current member.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from types import MappingProxyType
from typing import Any

from hedge_fund.domain.backtest import MemberSpan
from hedge_fund.universe import Universe

_HEADER = ["ticker", "start", "end"]


@dataclass(frozen=True)
class PointInTimeUniverse:
    """The index members on any date, as the Universe a cycle on that date uses.

    ``sectors`` and ``ciks`` are the frozen store's maps. A member without a sector or CIK
    is left out of that map only: build_snapshot drops it as no_sector and the fundamentals
    build marks it no_cik, so the cycle report counts it.
    """

    spans: tuple[MemberSpan, ...]
    sectors: Mapping[str, str]
    ciks: Mapping[str, str]
    name: str = "sp500_pit"

    def tickers_on(self, on: date) -> tuple[str, ...]:
        """Sorted tickers with a span containing ``on``."""
        return tuple(sorted({span.ticker for span in self.spans if span.contains(on)}))

    def members(self, on: date) -> Universe:
        """The members on ``on`` as a Universe with ``list_date`` ``on``, tickers sorted."""
        tickers = self.tickers_on(on)
        return Universe(
            name=self.name,
            list_date=on,
            tickers=tickers,
            sectors=MappingProxyType({t: self.sectors[t] for t in tickers if t in self.sectors}),
            ciks=MappingProxyType({t: self.ciks[t] for t in tickers if t in self.ciks}),
        )


def ever_members(spans: Sequence[MemberSpan], first: date, last: date) -> tuple[str, ...]:
    """Sorted tickers that are members on at least one day of ``first..last`` inclusive."""
    return tuple(sorted({span.ticker for span in spans if span.overlaps(first, last)}))


def parse_membership(text: str) -> tuple[MemberSpan, ...]:
    """Spans from the membership text, sorted by start, then ticker.

    Raises ValueError on a bad header, a short or long row, an empty ticker, a date that is
    not YYYY-MM-DD, an empty span or two overlapping spans of one ticker.
    """
    body = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    reader = csv.DictReader(io.StringIO(body))
    if reader.fieldnames != _HEADER:
        raise ValueError("membership header must be 'ticker,start,end'")
    spans = [_span(row, number) for number, row in enumerate(reader, start=1)]
    _check_overlaps(spans)
    return tuple(sorted(spans, key=lambda span: (span.start, span.ticker)))


def format_membership(spans: Sequence[MemberSpan], provenance: Sequence[str]) -> str:
    """The text parse_membership reads: '# ' + each provenance line, the header, the spans."""
    lines = [f"# {line}" for line in provenance] + [",".join(_HEADER)]
    for span in spans:
        end = "" if span.end is None else span.end.isoformat()
        lines.append(f"{span.ticker},{span.start.isoformat()},{end}")
    return "\n".join(lines) + "\n"


def spans_from_compositions(
    compositions: Sequence[tuple[date, frozenset[str]]],
) -> tuple[MemberSpan, ...]:
    """Spans from dated full member lists: each list holds from its date to the next one's.

    A ticker in consecutive lists keeps one span; one that leaves and returns gets two. The
    last list's members stay open. Sorted by start, then ticker.

    Raises ValueError when the dates are not strictly increasing.
    """
    if any(later <= earlier for (earlier, _), (later, _) in pairwise(compositions)):
        raise ValueError("composition dates must be strictly increasing")
    opened: dict[str, date] = {}
    spans: list[MemberSpan] = []
    for day, members in compositions:
        for ticker in sorted(set(opened) - members):
            spans.append(MemberSpan(ticker, opened.pop(ticker), day))
        for ticker in sorted(members - set(opened)):
            opened[ticker] = day
    spans += [MemberSpan(ticker, start, None) for ticker, start in opened.items()]
    return tuple(sorted(spans, key=lambda span: (span.start, span.ticker)))


def _span(row: Mapping[Any, Any], number: int) -> MemberSpan:
    where = f"membership row {number}"
    if any(row.get(column) is None for column in _HEADER):
        raise ValueError(f"{where} is missing a column")
    if row.get(None):
        raise ValueError(f"{where} has an extra field")
    ticker = str(row["ticker"]).strip()
    if not ticker:
        raise ValueError(f"{where} has an empty ticker")
    raw_end = str(row["end"]).strip()
    try:
        start = date.fromisoformat(str(row["start"]).strip())
        end = date.fromisoformat(raw_end) if raw_end else None
    except ValueError:
        raise ValueError(f"{where}: dates must be YYYY-MM-DD") from None
    return MemberSpan(ticker=ticker, start=start, end=end)


def _check_overlaps(spans: Sequence[MemberSpan]) -> None:
    by_ticker: dict[str, list[MemberSpan]] = {}
    for span in spans:
        by_ticker.setdefault(span.ticker, []).append(span)
    for ticker, own in by_ticker.items():
        ordered = sorted(own, key=lambda span: span.start)
        for earlier, later in pairwise(ordered):
            if earlier.end is None or later.start < earlier.end:
                raise ValueError(f"membership: overlapping spans for {ticker}")
