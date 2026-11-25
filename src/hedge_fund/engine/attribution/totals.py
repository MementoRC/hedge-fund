"""Totals of attributed day rows by ticker, by sector and overall. Pure."""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable, Sequence

from hedge_fund.domain.attribution import AttributionTotals, DayAttribution


def aggregate(
    days: Iterable[DayAttribution], excluded: Collection[str]
) -> tuple[dict[str, AttributionTotals], dict[str, AttributionTotals], AttributionTotals]:
    """(by ticker, by sector, overall) over the rows whose ticker is not in ``excluded``.

    Keys are sorted; a row's ``sector`` is its bucket (``"Unknown"`` included). Sums use
    ``math.fsum``; ``total`` sums every component of every row.
    """
    kept = [row for row in days if row.ticker not in excluded]
    by_ticker: dict[str, list[DayAttribution]] = {}
    by_sector: dict[str, list[DayAttribution]] = {}
    for row in kept:
        by_ticker.setdefault(row.ticker, []).append(row)
        by_sector.setdefault(row.sector, []).append(row)
    return (
        {ticker: _totals(rows) for ticker, rows in sorted(by_ticker.items())},
        {sector: _totals(rows) for sector, rows in sorted(by_sector.items())},
        _totals(kept),
    )


def _totals(rows: Sequence[DayAttribution]) -> AttributionTotals:
    return AttributionTotals(
        market=math.fsum(row.market for row in rows),
        sector=math.fsum(row.sector_pnl for row in rows),
        residual=math.fsum(row.residual for row in rows),
        trading=math.fsum(row.trading for row in rows),
        total=math.fsum(
            part for row in rows for part in (row.market, row.sector_pnl, row.residual, row.trading)
        ),
    )
