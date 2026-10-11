"""Per-ticker parquet cache decorator for any FundamentalsProvider."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from hedge_fund.adapters.data._atomic import atomic_write
from hedge_fund.adapters.data.caching import _SAFE_TICKER
from hedge_fund.engine.fundamentals.metrics import TAGS_HASH
from hedge_fund.ports.fundamentals import FundamentalsProvider, coerce_facts, empty_facts

DEFAULT_TTL = timedelta(days=1)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class CachingFundamentalsProvider:
    """Serves facts from cache_dir/fundamentals, refetching a ticker's whole history when stale.

    Each ticker has fundamentals/<TICKER>.parquet (its facts) plus <TICKER>.json holding
    {"fetched": ISO time, "tags_hash": hash of the curated tag list}. An entry is fresh while
    clock() - fetched < ttl and its tags_hash matches TAGS_HASH; otherwise the whole entry is
    refetched and replaced. EDGAR facts are append-only, so there is no date-range logic.
    Empty results are not cached. Unreadable entries count as misses and are overwritten.
    Duplicate CIKs across tickers (GOOG/GOOGL) simply get one entry per ticker.
    """

    def __init__(
        self,
        inner: FundamentalsProvider,
        cache_dir: Path,
        clock: Callable[[], datetime] = _utc_now,
        ttl: timedelta = DEFAULT_TTL,
    ) -> None:
        self._inner = inner
        self._dir = Path(cache_dir) / "fundamentals"
        self._clock = clock
        self._ttl = ttl

    def facts(self, tickers: Sequence[str]) -> pd.DataFrame:
        requested = list(dict.fromkeys(tickers))
        served: dict[str, pd.DataFrame] = {}
        missing: list[str] = []
        for ticker in requested:
            cached = self._read(ticker)
            if cached is None:
                missing.append(ticker)
            else:
                served[ticker] = cached
        if missing:
            served.update(self._refresh(missing))
        frames = [served[ticker] for ticker in requested if not served[ticker].empty]
        if not frames:
            return empty_facts()
        return coerce_facts(pd.concat(frames, ignore_index=True))

    def _refresh(self, tickers: list[str]) -> dict[str, pd.DataFrame]:
        fetched = self._inner.facts(tickers)
        fetched_at = self._clock()
        fresh: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            rows = coerce_facts(fetched.loc[fetched["ticker"] == ticker])
            if not rows.empty:
                self._write(ticker, rows, fetched_at)
            fresh[ticker] = rows
        return fresh

    def _paths(self, ticker: str) -> tuple[Path, Path]:
        if not _SAFE_TICKER.fullmatch(ticker):
            raise ValueError(f"ticker not safe as a cache file name: {ticker!r}")
        return self._dir / f"{ticker}.parquet", self._dir / f"{ticker}.json"

    def _read(self, ticker: str) -> pd.DataFrame | None:
        parquet, meta_path = self._paths(ticker)
        if not (parquet.is_file() and meta_path.is_file()):
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            fetched = datetime.fromisoformat(meta["fetched"])
            if meta["tags_hash"] != TAGS_HASH or self._clock() - fetched >= self._ttl:
                return None
            return coerce_facts(pd.read_parquet(parquet))
        except (OSError, ValueError, KeyError, TypeError):
            return None  # corrupt cache files count as a miss and are overwritten on refetch

    def _write(self, ticker: str, rows: pd.DataFrame, fetched_at: datetime) -> None:
        parquet, meta_path = self._paths(ticker)
        atomic_write(parquet, rows.to_parquet)
        body = json.dumps({"fetched": fetched_at.isoformat(), "tags_hash": TAGS_HASH})
        atomic_write(meta_path, lambda path: path.write_text(body, encoding="utf-8"))
