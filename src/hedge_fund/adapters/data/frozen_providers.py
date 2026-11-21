"""DataProvider, FundamentalsProvider and MacroProvider over a loaded frozen store (offline, point in time)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pandas as pd

from hedge_fund.domain.backtest import FrozenData
from hedge_fund.domain.ledger import Split
from hedge_fund.engine.backtest.pit_prices import closes_as_of
from hedge_fund.ports.data import DataProviderError
from hedge_fund.ports.fundamentals import coerce_facts
from hedge_fund.ports.macro import MacroProviderError


class FrozenDataProvider:
    """The answers a vendor would have given on ``as_of``, from a loaded frozen store.

    ``as_of`` is fixed per instance: run_cycle reads this port only through windows ending
    at its as_of, so the backtest builds one provider per cycle date with ``with_as_of``,
    which shares the loaded store. Closes come from engine.backtest.pit_prices.closes_as_of;
    a closes or splits request ending after ``as_of`` raises DataProviderError rather than
    being truncated. Sectors are the store's (current, not historical); an unknown ticker
    is an all-NaN column and a None sector.
    """

    def __init__(self, data: FrozenData, as_of: date) -> None:
        if not data.start <= as_of <= data.end:
            raise ValueError(f"as_of {as_of} is outside the store's {data.start}..{data.end}")
        self._data = data
        self._as_of = as_of

    @property
    def as_of(self) -> date:
        """The date this provider answers as of."""
        return self._as_of

    def with_as_of(self, as_of: date) -> FrozenDataProvider:
        """A provider for another date over the same loaded store (nothing is copied)."""
        return FrozenDataProvider(self._data, as_of)

    def closes(self, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        self._check_end("closes", end)
        stored = self._data.closes.index
        index = stored[(stored >= pd.Timestamp(start)) & (stored <= pd.Timestamp(end))]
        columns = {ticker: self._seen(ticker).reindex(index) for ticker in dict.fromkeys(tickers)}
        frame = pd.DataFrame(columns, index=index, dtype="float64")
        return frame.reindex(columns=list(tickers))

    def splits(self, tickers: Sequence[str], start: date, end: date) -> tuple[Split, ...]:
        self._check_end("splits", end)
        order = {ticker: position for position, ticker in enumerate(dict.fromkeys(tickers))}
        found = [
            split
            for split in self._data.splits
            if split.ticker in order and start < split.ex_date <= end
        ]
        return tuple(sorted(found, key=lambda split: (split.ex_date, order[split.ticker])))

    def sectors(self, tickers: Sequence[str]) -> dict[str, str | None]:
        return {ticker: self._data.sectors.get(ticker) for ticker in tickers}

    def _seen(self, ticker: str) -> pd.Series:
        data = self._data
        if ticker not in data.closes.columns:
            return pd.Series(dtype="float64", index=pd.DatetimeIndex([]))
        return closes_as_of(data.closes[ticker], ticker, data.splits, data.dividends, self._as_of)

    def _check_end(self, method: str, end: date) -> None:
        if end > self._as_of:
            raise DataProviderError(
                f"{method} end {end} is after as_of {self._as_of}: the frozen store never "
                "serves data from after the cycle date"
            )


class FrozenFundamentalsProvider:
    """Every stored fact row for the requested tickers, restatements included, unfiltered.

    The port has no as_of: point-in-time selection is derive.py's (pit_filter keeps
    ``filed < as_of``, select_versions the latest filing), so this provider never drops
    or deduplicates a row.
    """

    def __init__(self, data: FrozenData) -> None:
        self._facts = coerce_facts(data.facts)

    def facts(self, tickers: Sequence[str]) -> pd.DataFrame:
        return coerce_facts(self._facts.loc[self._facts["ticker"].isin(list(tickers))])


class FrozenMacroProvider:
    """The macro series a vendor would have served on ``as_of``, from a loaded frozen store.

    One instance per cycle date, like FrozenDataProvider. Rows keep their raw observation
    dates: the one-day point-in-time cut is the engine's (engine.factors.macro). A request
    ending after ``as_of`` or naming a series the store lacks raises MacroProviderError.
    """

    def __init__(self, data: FrozenData, as_of: date) -> None:
        if not data.start <= as_of <= data.end:
            raise ValueError(f"as_of {as_of} is outside the store's {data.start}..{data.end}")
        self._macro = data.macro
        self._as_of = as_of

    def series(self, ids: Sequence[str], start: date, end: date) -> pd.DataFrame:
        if end > self._as_of:
            raise MacroProviderError(
                f"macro end {end} is after as_of {self._as_of}: the frozen store never serves "
                "data from after the cycle date"
            )
        missing = [series_id for series_id in ids if series_id not in self._macro.columns]
        if missing:
            raise MacroProviderError(f"the frozen store has no macro series {missing}")
        stored = self._macro.index
        rows = (stored >= pd.Timestamp(start)) & (stored <= pd.Timestamp(end))
        return self._macro.loc[rows, list(ids)]
