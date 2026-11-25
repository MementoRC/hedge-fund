"""Tests for the yfinance history source (offline: the Ticker factory is injected)."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import pytest

from hedge_fund.adapters.data.yfinance_history import HISTORY_COLUMNS, YFinanceHistory
from hedge_fund.ports.data import DataProviderError

STAMPS = ["2024-06-06", "2024-06-07", "2024-06-10", "2024-06-11"]


class FakeTicker:
    """Stand-in for yf.Ticker: ``FakeTicker(frame)(symbol).history(**kwargs)`` records calls."""

    def __init__(self, frame: pd.DataFrame | None = None, error: Exception | None = None) -> None:
        self.frame = frame
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._symbol = ""

    def __call__(self, symbol: str) -> FakeTicker:
        self._symbol = symbol
        return self

    def history(self, **kwargs: Any) -> pd.DataFrame | None:
        self.calls.append((self._symbol, kwargs))
        if self.error is not None:
            raise self.error
        return self.frame


def _yf_frame(stamps: list[str] = STAMPS) -> pd.DataFrame:
    index = pd.DatetimeIndex(stamps).tz_localize("America/New_York")
    n = len(stamps)
    return pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(n)],
            "High": [110.0] * n,
            "Low": [90.0] * n,
            "Close": [101.0 + i for i in range(n)],
            "Adj Close": [100.5 + i for i in range(n)],
            "Volume": [1000] * n,
            "Dividends": [0.0] * (n - 1) + [0.01],
            "Stock Splits": [0.0] * n,
        },
        index=index,
    )


def _source(fake: FakeTicker, today: date = date(2024, 6, 12)) -> YFinanceHistory:
    return YFinanceHistory(ticker=fake, clock=lambda: today)


def test_keeps_open_close_and_dividends_on_naive_dates() -> None:
    frame = _source(FakeTicker(_yf_frame()))("NVDA", date(2024, 6, 6), date(2024, 6, 11))
    assert frame is not None
    assert tuple(frame.columns) == HISTORY_COLUMNS
    assert list(frame.index) == [pd.Timestamp(stamp) for stamp in STAMPS]
    assert frame.index.tz is None
    assert frame["Dividends"].tolist() == [0.0, 0.0, 0.0, 0.01]
    assert (frame.dtypes == "float64").all()


def test_requests_end_plus_one_day_unadjusted_with_actions() -> None:
    fake = FakeTicker(_yf_frame())
    _source(fake)("NVDA", date(2024, 6, 6), date(2024, 6, 11))
    assert fake.calls == [
        (
            "NVDA",
            {
                "start": date(2024, 6, 6),
                "end": date(2024, 6, 12),
                "auto_adjust": False,
                "actions": True,
            },
        )
    ]


def test_bars_on_or_after_today_are_dropped() -> None:
    frame = _source(FakeTicker(_yf_frame()), today=date(2024, 6, 11))(
        "NVDA", date(2024, 6, 6), date(2024, 6, 11)
    )
    assert frame is not None
    assert frame.index[-1] == pd.Timestamp("2024-06-10")


def test_only_bars_from_today_give_none() -> None:
    source = _source(FakeTicker(_yf_frame(["2024-06-12"])))
    assert source("NVDA", date(2024, 6, 12), date(2024, 6, 12)) is None


def test_duplicate_dates_keep_the_last_bar() -> None:
    raw = _yf_frame(["2024-06-06", "2024-06-06"])
    frame = _source(FakeTicker(raw))("NVDA", date(2024, 6, 6), date(2024, 6, 6))
    assert frame is not None
    assert frame["Close"].tolist() == [102.0]


@pytest.mark.parametrize("raw", [None, pd.DataFrame()])
def test_no_data_gives_none(raw: pd.DataFrame | None) -> None:
    assert _source(FakeTicker(raw))("GONE", date(2024, 6, 6), date(2024, 6, 11)) is None


def test_a_vendor_failure_raises_data_provider_error() -> None:
    boom = RuntimeError("rate limited")
    with pytest.raises(DataProviderError, match="history failed for NVDA") as info:
        _source(FakeTicker(error=boom))("NVDA", date(2024, 6, 6), date(2024, 6, 11))
    assert info.value.__cause__ is boom


def test_a_frame_without_the_columns_raises() -> None:
    raw = _yf_frame().drop(columns=["Dividends"])
    with pytest.raises(DataProviderError, match="lacks"):
        _source(FakeTicker(raw))("NVDA", date(2024, 6, 6), date(2024, 6, 11))
