"""Plain data for the backtest's frozen store."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import pandas as pd

from hedge_fund.domain.ledger import Split


@dataclass(frozen=True)
class Dividend:
    """A cash dividend of ``amount`` per share, ex-dated ``ex_date``.

    ``amount`` is on the frozen store's split basis (every split up to the freeze date
    applied), the basis of the stored Open and Close.
    """

    ticker: str
    ex_date: date
    amount: float


@dataclass(frozen=True)
class MemberSpan:
    """``ticker`` is an index member on every day ``d`` with ``start <= d < end``.

    ``end`` None means still a member. Each date is the first dataset snapshot on or after
    the effective date, half-open ``[start, end)``: a name is a member on ``start`` and not
    on ``end``.
    """

    ticker: str
    start: date
    end: date | None

    def __post_init__(self) -> None:
        if self.end is not None and self.end <= self.start:
            raise ValueError(f"{self.ticker}: span end {self.end} is not after start {self.start}")

    def contains(self, day: date) -> bool:
        """True when ``ticker`` is a member on ``day``."""
        return self.start <= day and (self.end is None or day < self.end)

    def overlaps(self, first: date, last: date) -> bool:
        """True when the span holds at least one day of ``first..last`` inclusive."""
        return self.start <= last and (self.end is None or self.end > first)


@dataclass(frozen=True)
class FlaggedSpan:
    """A membership span of a reused ticker that belongs to the wrong company.

    ``span`` is half-open ``[start, end)`` like MemberSpan; ``reason`` names the company.
    """

    span: MemberSpan
    reason: str


@dataclass(frozen=True, eq=False)
class FrozenData:
    """Everything one offline backtest reads, frozen once by ``backtest snapshot``.

    ``opens`` and ``closes`` are dates x tickers (priced tickers only, SPY included) on one
    tz-naive, sorted, unique DatetimeIndex (SPY's trading days) within ``start..end``, with
    yfinance ``auto_adjust=False`` values: split-adjusted to the freeze date, not
    dividend-adjusted. ``splits`` is the priced tickers' full split history through the
    freeze date (later than ``end`` included: the stored prices are adjusted for them);
    ``dividends`` those ex-dated in ``start..end``. ``facts`` is the raw FACT_COLUMNS frame,
    every filing's row. ``unpriced``, ``no_sector``, ``no_cik`` and ``sic_sectors`` are
    sorted member tickers: no price rows, no sector, no CIK, sector from the SEC SIC code.
    Frozen only prevents field reassignment; readers must never mutate the frames.
    """

    start: date
    end: date
    opens: pd.DataFrame
    closes: pd.DataFrame
    splits: tuple[Split, ...]
    dividends: tuple[Dividend, ...]
    facts: pd.DataFrame
    sectors: Mapping[str, str]
    ciks: Mapping[str, str]
    membership: tuple[MemberSpan, ...]
    unpriced: tuple[str, ...]
    no_sector: tuple[str, ...]
    no_cik: tuple[str, ...]
    sic_sectors: tuple[str, ...]

    def __post_init__(self) -> None:
        index = self.closes.index
        if not isinstance(index, pd.DatetimeIndex) or index.tz is not None:
            raise ValueError("closes index must be a tz-naive DatetimeIndex")
        if not index.is_monotonic_increasing or index.has_duplicates:
            raise ValueError("closes index must be sorted and unique")
        same_tickers = list(self.opens.columns) == list(self.closes.columns)
        if not (same_tickers and self.opens.index.equals(index)):
            raise ValueError("opens and closes must share dates and tickers")
        if len(index) and (index[0].date() < self.start or index[-1].date() > self.end):
            raise ValueError(f"prices fall outside {self.start}..{self.end}")
