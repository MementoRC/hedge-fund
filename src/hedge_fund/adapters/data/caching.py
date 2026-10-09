"""Per-ticker parquet cache decorator for any DataProvider."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from hedge_fund.adapters.data._atomic import atomic_write
from hedge_fund.ports.data import DataProvider

_SAFE_TICKER = re.compile(r"[A-Za-z0-9^=_-][A-Za-z0-9.^=_-]*")
_ONE_DAY = timedelta(days=1)


@dataclass(frozen=True)
class _CachedSeries:
    closes: pd.Series
    start: date
    end: date

    def covers(self, start: date, end: date) -> bool:
        return self.start <= start and end <= self.end


class CachingProvider:
    """Serves closes from cache_dir, refetching from `inner` only when needed.

    Each ticker has prices/<TICKER>.parquet plus prices/<TICKER>.json recording the
    covered range. A request the cache does not fully cover is refetched over the union
    of the requested and cached ranges and replaces the cache: adjusted closes are
    restated after dividends and splits, so fetches are never spliced together.
    sectors.json caches resolved (non-None) sectors only, so empty vendor answers are retried.
    """

    def __init__(
        self,
        inner: DataProvider,
        cache_dir: Path,
        clock: Callable[[], date] = date.today,
    ) -> None:
        self._inner = inner
        self._prices_dir = Path(cache_dir) / "prices"
        self._sectors_path = Path(cache_dir) / "sectors.json"
        self._clock = clock

    def closes(self, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        requested = list(tickers)
        cached = {ticker: self._read(ticker) for ticker in requested}
        served: dict[str, pd.Series] = {}
        groups: dict[tuple[date, date], list[str]] = {}
        for ticker, entry in cached.items():
            if entry is None:
                groups.setdefault((start, end), []).append(ticker)
            elif entry.covers(start, end):
                served[ticker] = entry.closes
            else:
                union = (min(start, entry.start), max(end, entry.end))
                groups.setdefault(union, []).append(ticker)
        for (fetch_start, fetch_end), group in groups.items():
            fetched = self._inner.closes(group, fetch_start, fetch_end)
            covered_end = min(fetch_end, self._clock() - _ONE_DAY)
            for ticker in group:
                fresh = fetched[ticker].dropna()
                if fresh.empty:
                    entry = cached[ticker]
                    served[ticker] = entry.closes if entry is not None else _empty_series()
                    continue
                if covered_end >= fetch_start:
                    self._write(ticker, fresh, fetch_start, covered_end)
                served[ticker] = fresh
        return _assemble(requested, served, start, end)

    def sectors(self, tickers: Sequence[str]) -> dict[str, str | None]:
        requested = list(tickers)
        known = self._read_sectors()
        missing = [ticker for ticker in requested if ticker not in known]
        if missing:
            resolved = {t: s for t, s in self._inner.sectors(missing).items() if s is not None}
            if resolved:
                known.update(resolved)
                body = json.dumps(known, indent=2, sort_keys=True)
                atomic_write(
                    self._sectors_path, lambda path: path.write_text(body, encoding="utf-8")
                )
        return {ticker: known.get(ticker) for ticker in requested}

    def _read_sectors(self) -> dict[str, str]:
        if not self._sectors_path.is_file():
            return {}
        known: dict[str, str] = json.loads(self._sectors_path.read_text(encoding="utf-8"))
        return known

    def _paths(self, ticker: str) -> tuple[Path, Path]:
        if not _SAFE_TICKER.fullmatch(ticker):
            raise ValueError(f"ticker not safe as a cache file name: {ticker!r}")
        return self._prices_dir / f"{ticker}.parquet", self._prices_dir / f"{ticker}.json"

    def _read(self, ticker: str) -> _CachedSeries | None:
        parquet, coverage = self._paths(ticker)
        if not (parquet.is_file() and coverage.is_file()):
            return None
        span = json.loads(coverage.read_text(encoding="utf-8"))
        return _CachedSeries(
            closes=pd.read_parquet(parquet)["close"],
            start=date.fromisoformat(span["start"]),
            end=date.fromisoformat(span["end"]),
        )

    def _write(self, ticker: str, closes: pd.Series, start: date, end: date) -> None:
        parquet, coverage = self._paths(ticker)
        atomic_write(parquet, closes.rename("close").to_frame().to_parquet)
        span = json.dumps({"start": start.isoformat(), "end": end.isoformat()})
        atomic_write(coverage, lambda path: path.write_text(span, encoding="utf-8"))


def _empty_series() -> pd.Series:
    return pd.Series(dtype="float64", index=pd.DatetimeIndex([]))


def _assemble(
    tickers: list[str], served: dict[str, pd.Series], start: date, end: date
) -> pd.DataFrame:
    """Align served series on their union of dates, in request order, sliced to start..end."""
    index = pd.DatetimeIndex([])
    for series in served.values():
        index = index.union(pd.DatetimeIndex(series.index))
    frame = pd.DataFrame(
        {ticker: served[ticker].reindex(index) for ticker in tickers},
        index=index,
        columns=tickers,
    )
    in_range = (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
    return frame.loc[in_range]
