"""Realised attribution types (Layer 7b): one ticker's P&L on one trading day split into
market, sector, residual and trading, their totals, and the attribution report.

None of these types validate themselves.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum

from hedge_fund.domain.ledger import ReconciliationBreak

UNKNOWN_SECTOR = "Unknown"
DIVIDEND_NOTE = (
    "closes are dividend-adjusted and fill prices are not, so dollar P&L drifts slightly "
    "around ex-dividend dates"
)


class AttributeOutcome(StrEnum):
    """How one attribution ended: BREAK when the ledger and the broker disagree; FAILED is
    the partial report of a run that raised.
    """

    CLEAN = "clean"
    BREAK = "break"
    FAILED = "failed"


@dataclass(frozen=True)
class DayAttribution:
    """One ticker on one trading ``day``.

    ``prior_shares`` is the signed holding at the previous close. ``market + sector_pnl +
    residual`` is the holding P&L ``prior_shares * (P_t - P_{t-1})``; ``trading`` is the
    day's fills marked to the close, ``sum q * (P_t - price)``. ``regressed`` is False when
    there was no holding or the holding P&L was attributed all-residual. ``sector`` is
    UNKNOWN_SECTOR when the provider has none.
    """

    day: date
    ticker: str
    sector: str
    prior_shares: int
    market: float
    sector_pnl: float
    residual: float
    trading: float
    regressed: bool


@dataclass(frozen=True)
class AttributionTotals:
    """Summed components; ``total = market + sector + residual + trading``."""

    market: float
    sector: float
    residual: float
    trading: float
    total: float


NO_PNL = AttributionTotals(market=0.0, sector=0.0, residual=0.0, trading=0.0, total=0.0)


@dataclass(frozen=True)
class AttributionReport:
    """Audit record of one attribution; fields not reached are empty, totals NO_PNL.

    ``start`` / ``end`` are the window's bounds (None only on a FAILED run that had not
    resolved them). ``started_at`` is tz-aware. ``failed_stage`` and ``error``
    (``"<Type>: <message>"``) are set only for FAILED. ``days`` holds every row, including
    those of break and excluded tickers; ``by_ticker``, ``by_sector`` (UNKNOWN_SECTOR is a
    bucket) and ``total`` leave out the tickers in ``breaks`` and ``excluded`` (ticker ->
    reason). ``no_beta`` maps a ticker to the days it was held but not regressed while the
    day's regression ran; ``unregressed_days`` are the days with a holding whose regression
    could not run (all holdings attributed residual). DIVIDEND_NOTE applies to every figure.
    """

    start: date | None
    end: date | None
    started_at: datetime
    outcome: AttributeOutcome
    failed_stage: str | None
    error: str | None
    days: tuple[DayAttribution, ...]
    by_ticker: Mapping[str, AttributionTotals]
    by_sector: Mapping[str, AttributionTotals]
    total: AttributionTotals
    breaks: tuple[ReconciliationBreak, ...]
    no_beta: Mapping[str, tuple[date, ...]]
    unregressed_days: tuple[date, ...]
    excluded: Mapping[str, str]
