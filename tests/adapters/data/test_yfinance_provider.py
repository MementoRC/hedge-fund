"""Tests for the yfinance DataProvider adapter (offline: vendor calls are injected)."""

from collections.abc import Mapping
from datetime import date, timedelta
from typing import Any

import pandas as pd
import pytest
import yfinance as yf

from hedge_fund.adapters.data.yfinance_provider import YAHOO_TO_GICS, YFinanceProvider
from hedge_fund.ports.data import DataProviderError
from hedge_fund.universe import GICS_SECTORS
from tests.factories import make_closes
from tests.fakes import RecordingDownload, assert_closes_equal, yf_multi, yf_single

START = date(2026, 9, 1)
END = date(2026, 9, 30)


def test_closes_normalizes_multi_ticker_frame() -> None:
    closes = make_closes(["AAA", "BBB"], n_days=21)
    provider = YFinanceProvider(download=RecordingDownload(yf_multi(closes)))
    assert_closes_equal(provider.closes(["AAA", "BBB"], START, END), closes)


def test_closes_requests_end_plus_one_day_with_adjusted_prices() -> None:
    download = RecordingDownload(yf_multi(make_closes(["AAA"], n_days=5)))
    YFinanceProvider(download=download).closes(["AAA"], START, END)
    assert download.calls == [
        (
            ["AAA"],
            {
                "start": START,
                "end": END + timedelta(days=1),
                "auto_adjust": True,
                "actions": False,
                "progress": False,
            },
        )
    ]


def test_closes_normalizes_flat_single_ticker_frame() -> None:
    closes = make_closes(["AAA"], n_days=21)
    provider = YFinanceProvider(download=RecordingDownload(yf_single(closes["AAA"])))
    assert_closes_equal(provider.closes(["AAA"], START, END), closes)


def test_closes_rejects_flat_frame_for_several_tickers() -> None:
    closes = make_closes(["AAA"], n_days=5)
    provider = YFinanceProvider(download=RecordingDownload(yf_single(closes["AAA"])))
    with pytest.raises(DataProviderError, match="flat"):
        provider.closes(["AAA", "BBB"], START, END)


def test_closes_missing_ticker_is_nan_column() -> None:
    closes = make_closes(["AAA"], n_days=21)
    result = YFinanceProvider(download=RecordingDownload(yf_multi(closes))).closes(
        ["ZZZ", "AAA"], START, END
    )
    assert list(result.columns) == ["ZZZ", "AAA"]
    assert result["ZZZ"].isna().all()
    assert_closes_equal(result[["AAA"]], closes)


def test_closes_empty_download_gives_empty_frame_with_requested_columns() -> None:
    provider = YFinanceProvider(download=RecordingDownload(pd.DataFrame()))
    result = provider.closes(["AAA", "BBB"], START, END)
    assert list(result.columns) == ["AAA", "BBB"]
    assert result.empty
    assert isinstance(result.index, pd.DatetimeIndex)


def test_closes_with_no_tickers_skips_download() -> None:
    download = RecordingDownload()
    result = YFinanceProvider(download=download).closes([], START, END)
    assert download.calls == []
    assert result.empty
    assert isinstance(result.index, pd.DatetimeIndex)


def test_closes_strips_timezone_and_sorts() -> None:
    closes = make_closes(["AAA"], n_days=5)
    reversed_rows = closes.iloc[::-1]
    aware = reversed_rows.set_axis(reversed_rows.index.tz_localize("America/New_York"))
    result = YFinanceProvider(download=RecordingDownload(yf_multi(aware))).closes(
        ["AAA"], START, END
    )
    assert result.index.tz is None
    assert_closes_equal(result, closes)


def test_closes_drops_bars_on_or_after_clock_today() -> None:
    today = date(2026, 9, 10)
    days = pd.DatetimeIndex(["2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11"])
    frame = pd.DataFrame({"AAA": [1.0, 2.0, 3.0, 4.0]}, index=days)
    provider = YFinanceProvider(download=RecordingDownload(yf_multi(frame)), clock=lambda: today)
    result = provider.closes(["AAA"], date(2026, 9, 1), today)
    assert_closes_equal(result, frame.iloc[:2])


def test_closes_wraps_vendor_exception() -> None:
    boom = RuntimeError("rate limited")
    provider = YFinanceProvider(download=RecordingDownload(error=boom))
    with pytest.raises(DataProviderError) as info:
        provider.closes(["AAA"], START, END)
    assert info.value.__cause__ is boom


def test_sectors_maps_yahoo_names_to_gics() -> None:
    infos: dict[str, Mapping[str, Any] | None] = {
        "AAA": {"sector": "Technology"},
        "BBB": {"sector": "Financial Services"},
        "CCC": {"sector": "Crypto"},
        "DDD": {},
        "EEE": None,
    }
    provider = YFinanceProvider(info_fetcher=infos.__getitem__)
    assert provider.sectors(["AAA", "BBB", "CCC", "DDD", "EEE"]) == {
        "AAA": "Information Technology",
        "BBB": "Financials",
        "CCC": None,
        "DDD": None,
        "EEE": None,
    }


def test_sectors_wraps_vendor_exception() -> None:
    boom = RuntimeError("404")

    def failing(ticker: str) -> Mapping[str, Any]:
        raise boom

    with pytest.raises(DataProviderError) as info:
        YFinanceProvider(info_fetcher=failing).sectors(["AAA"])
    assert info.value.__cause__ is boom


def test_yahoo_to_gics_covers_every_gics_sector() -> None:
    assert len(YAHOO_TO_GICS) == 11
    assert set(YAHOO_TO_GICS.values()) == GICS_SECTORS


def test_default_download_is_yfinance() -> None:
    assert YFinanceProvider()._download is yf.download
