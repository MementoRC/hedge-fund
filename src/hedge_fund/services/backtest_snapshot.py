"""Build the frozen backtest data online, once, from injected vendor sources.

Services never import adapters: the CLI wires yfinance and SEC EDGAR into SnapshotSources
and writes the returned FrozenData with adapters/data/frozen_store.py.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

from hedge_fund.domain.backtest import Dividend, FlaggedSpan, FrozenData, MemberSpan
from hedge_fund.domain.ledger import Split
from hedge_fund.engine.backtest.membership import ever_members
from hedge_fund.engine.backtest.sic import sic_sector
from hedge_fund.ports.fundamentals import FundamentalsProvider, coerce_facts, empty_facts

BENCHMARK = "SPY"

PriceHistory = Callable[[str, date, date], pd.DataFrame | None]
SplitHistory = Callable[[Sequence[str]], tuple[Split, ...]]
FundamentalsFactory = Callable[[Mapping[str, str]], FundamentalsProvider]


@dataclass(frozen=True)
class SnapshotSources:
    """The vendors one snapshot reads.

    ``history(ticker, start, end)``: daily Open, Close and Dividends (yfinance
    ``auto_adjust=False``) on tz-naive dates, or None. ``splits(tickers)``: every split of
    those tickers through the fetch date. ``company_ciks()``: current ticker -> CIK.
    ``sic(cik)``: the CIK's SIC code or None. ``fundamentals(ciks)``: a
    FundamentalsProvider for those CIKs.
    """

    history: PriceHistory
    splits: SplitHistory
    company_ciks: Callable[[], Mapping[str, str]]
    sic: Callable[[str], str | None]
    fundamentals: FundamentalsFactory


def build_frozen_data(
    sources: SnapshotSources,
    membership: Sequence[MemberSpan],
    *,
    sectors: Mapping[str, str],
    ciks: Mapping[str, str],
    start: date,
    end: date,
    flagged: Sequence[FlaggedSpan] = (),
) -> FrozenData:
    """Fetch everything the replay reads for every member in start..end, plus SPY.

    ``sectors`` and ``ciks`` are the static universe's. A member without a static CIK
    falls back to ``company_ciks()`` (current tickers only); one without a static sector
    falls back to its SIC code's sector when it has a CIK. A member with no close on SPY's
    trading days is unpriced, one with neither sector is no_sector, one with no CIK is
    no_cik and gets no facts; all are recorded. The store's end is SPY's last trading day.
    ``flagged`` is the reused-ticker spans; only those overlapping start..end are frozen.

    Raises ValueError when start > end or SPY has no prices (it is the store's calendar).
    """
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    members = ever_members(membership, start, end)
    histories = _histories([*members, BENCHMARK], sources.history, start, end)
    if BENCHMARK not in histories:
        raise ValueError(f"{BENCHMARK} has no prices in {start}..{end}: it is the calendar")
    opens, closes = _aligned(histories, pd.DatetimeIndex(histories[BENCHMARK].index))
    priced = [str(ticker) for ticker in closes.columns]
    member_ciks = _member_ciks(members, ciks, sources.company_ciks)
    member_sectors, from_sic = _member_sectors(members, sectors, member_ciks, sources.sic)
    return FrozenData(
        start=start,
        end=closes.index[-1].date(),
        opens=opens,
        closes=closes,
        splits=_splits(sources.splits(priced), priced),
        dividends=_dividends(histories, priced),
        facts=_facts(sources.fundamentals, member_ciks),
        sectors=member_sectors,
        ciks=member_ciks,
        membership=tuple(membership),
        unpriced=tuple(ticker for ticker in members if ticker not in priced),
        no_sector=tuple(ticker for ticker in members if ticker not in member_sectors),
        no_cik=tuple(ticker for ticker in members if ticker not in member_ciks),
        sic_sectors=from_sic,
        flagged=tuple(item for item in flagged if item.span.overlaps(start, end)),
    )


def _histories(
    tickers: Sequence[str], history: PriceHistory, start: date, end: date
) -> dict[str, pd.DataFrame]:
    """Each ticker's history trimmed to start..end; tickers with no close are left out."""
    found: dict[str, pd.DataFrame] = {}
    for ticker in dict.fromkeys(tickers):
        frame = history(ticker, start, end)
        if frame is None:
            continue
        index = pd.DatetimeIndex(frame.index)
        frame = frame.loc[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))]
        if frame["Close"].notna().any():
            found[ticker] = frame
    return found


def _aligned(
    histories: Mapping[str, pd.DataFrame], calendar: pd.DatetimeIndex
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Opens and closes on the SPY calendar, tickers sorted; no close there means dropped."""
    calendar = calendar.rename(None)
    closes = pd.DataFrame(
        {ticker: histories[ticker]["Close"].reindex(calendar) for ticker in sorted(histories)},
        index=calendar,
        dtype="float64",
    )
    closes = closes.loc[:, closes.notna().any()]
    opens = pd.DataFrame(
        {ticker: histories[ticker]["Open"].reindex(calendar) for ticker in closes.columns},
        index=calendar,
        dtype="float64",
    )
    return opens, closes


def _splits(splits: Sequence[Split], priced: Sequence[str]) -> tuple[Split, ...]:
    keep = set(priced)
    found = [split for split in splits if split.ticker in keep]
    return tuple(sorted(found, key=lambda split: (split.ex_date, split.ticker)))


def _dividends(
    histories: Mapping[str, pd.DataFrame], priced: Sequence[str]
) -> tuple[Dividend, ...]:
    """Non-zero Dividends entries of the priced tickers, by ex-date then ticker."""
    found = [
        Dividend(ticker, pd.Timestamp(stamp).date(), float(amount))
        for ticker in priced
        for stamp, amount in histories[ticker]["Dividends"].items()
        if pd.notna(amount) and amount > 0
    ]
    return tuple(sorted(found, key=lambda dividend: (dividend.ex_date, dividend.ticker)))


def _member_ciks(
    members: Sequence[str],
    ciks: Mapping[str, str],
    company_ciks: Callable[[], Mapping[str, str]],
) -> dict[str, str]:
    """The static CIK, else SEC's current one (fetched only when some member lacks one)."""
    current: Mapping[str, str] = (
        company_ciks() if any(ticker not in ciks for ticker in members) else {}
    )
    found: dict[str, str] = {}
    for ticker in members:
        cik = ciks.get(ticker) or current.get(ticker)
        if cik is not None:
            found[ticker] = cik
    return found


def _member_sectors(
    members: Sequence[str],
    sectors: Mapping[str, str],
    member_ciks: Mapping[str, str],
    sic: Callable[[str], str | None],
) -> tuple[dict[str, str], tuple[str, ...]]:
    """The static sector, else the SIC fallback's; plus the members sectored by SIC."""
    resolved: dict[str, str] = {}
    from_sic: list[str] = []
    for ticker in members:
        if ticker in sectors:
            resolved[ticker] = sectors[ticker]
            continue
        cik = member_ciks.get(ticker)
        sector = None if cik is None else sic_sector(sic(cik))
        if sector is not None:
            resolved[ticker] = sector
            from_sic.append(ticker)
    return resolved, tuple(from_sic)


def _facts(fundamentals: FundamentalsFactory, member_ciks: Mapping[str, str]) -> pd.DataFrame:
    if not member_ciks:
        return empty_facts()
    return coerce_facts(fundamentals(member_ciks).facts(list(member_ciks)))
