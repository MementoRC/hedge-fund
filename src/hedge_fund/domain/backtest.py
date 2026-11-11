"""Plain data for the backtest's frozen store."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

import pandas as pd

from hedge_fund.domain.ledger import Split

MEMBERSHIP_FLOOR = date(2008, 1, 1)  # decision D3: membership is reliable from here


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
    ``flagged`` is the reused-ticker spans excluded from the universe, frozen so a replay
    does not change when the packaged list is edited.
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
    flagged: tuple[FlaggedSpan, ...] = ()

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


@dataclass(frozen=True)
class SimCosts:
    """Simulated trading costs: adverse slippage on every fill and the short borrow rate.

    ``borrow_rate`` is per year on the short's market value, accrued ACT/365 over calendar
    days between closes.
    """

    slippage_bps: float = 5.0
    borrow_rate: float = 0.005

    def __post_init__(self) -> None:
        for name in ("slippage_bps", "borrow_rate"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0, got {value}")


class SimEventKind(StrEnum):
    FILL = "fill"
    REJECT = "reject"
    SPLIT = "split"
    DIVIDEND = "dividend"
    DELIST = "delist"
    BORROW = "borrow"


@dataclass(frozen=True)
class SimEvent:
    """One thing the simulated broker did on ``day``.

    The fields by kind:
    - ``shares`` is signed (+ bought, - sold) for FILL and DELIST, the share change for
      SPLIT, the holding for DIVIDEND and BORROW, and 0 for REJECT.
    - ``price`` is the raw fill price for FILL, the close for DELIST and BORROW, the
      per-share amount for DIVIDEND, and the open (cash in lieu) for SPLIT.
    - ``cash`` is the signed cash change.
    - ``cost`` is the slippage paid on a FILL and the borrow paid on a BORROW, 0 otherwise.
    - ``reason`` is set on REJECT only.
    """

    day: date
    kind: SimEventKind
    ticker: str
    shares: int
    price: float
    cash: float
    cost: float = 0.0
    client_order_id: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class Performance:
    """Return statistics of one equity curve (risk-free rate 0, 252 trading days a year).

    ``sharpe`` is None when ``volatility`` is 0. ``peak`` and ``trough`` bound the max
    drawdown (both the start date when there is none).
    """

    start: date
    end: date
    total_return: float
    cagr: float
    volatility: float
    sharpe: float | None
    max_drawdown: float
    peak: date
    trough: date


@dataclass(frozen=True)
class TradingCosts:
    """Trading volume and costs of a backtest, in dollars.

    ``turnover`` is two-sided and annualized: traded notional / mean NAV / years.
    """

    traded_notional: float
    turnover: float
    slippage: float
    borrow: float


@dataclass(frozen=True)
class Honesty:
    """What a backtest dropped, rejected, closed out or would have tripped.

    Ticker counts are distinct tickers over the whole run. ``price_unresolved`` and
    ``rejects`` are keyed by reason; ``outcomes`` by ``CycleOutcome`` value.
    """

    price_unresolved: Mapping[str, int]
    fundamentals_unresolved: int
    flagged_excluded: int
    stopped_trading: int
    rejects: Mapping[str, int]
    daily_breaker_days: int
    drawdown_breaker_days: int
    outcomes: Mapping[str, int]


@dataclass(frozen=True)
class BacktestMetrics:
    """The summary of one backtest run: performance vs the benchmark, costs, honesty."""

    strategy: Performance
    benchmark: Performance
    beta: float | None
    costs: TradingCosts
    honesty: Honesty


RESIDUAL_BIASES: tuple[str, ...] = (
    "Sectors are current, not historical.",
    "yfinance coverage of delisted names is incomplete (see the unpriced count).",
    "Membership accuracy depends on its source dataset.",
    "Late removals: dataset dates can trail the effective date, so a removed stock may "
    "be held a few extra days (late additions never look ahead).",
    "Ticker reuse not listed in the flagged-spans file may remain: a reused symbol can "
    "return another company's prices, CIK and facts.",
    "With monthly rebalances the breakers act only on the one-day move at cycle time; "
    "the would-trip breaker-day counts are the main evidence of breaker behaviour.",
)


@dataclass(frozen=True)
class BacktestReport:
    """The summary of one replay, written once the run completes."""

    run_id: str
    store_digest: str
    start: date
    end: date
    cash: float
    costs: SimCosts
    trading_days: int
    rebalances: int
    metrics: BacktestMetrics
    biases: tuple[str, ...] = RESIDUAL_BIASES
