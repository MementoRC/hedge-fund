"""Tests for the FRED fredgraph CSV adapter (offline: get and sleep injected)."""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import pandas as pd
import pytest
import requests

from hedge_fund.adapters.data.fred_provider import (
    FREDGRAPH_URL,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    FredProvider,
)
from hedge_fund.ports.macro import MacroProviderError

DGS10_CSV = (
    "observation_date,DGS10\n2024-01-01,.\n2024-01-02,3.95\n2024-01-03,3.91\n2024-01-04,3.99\n"
)
VIX_CSV = "observation_date,VIXCLS\n2024-01-02,13.20\n2024-01-03,14.04\n2024-01-05,13.35\n"
DGS10_URL = FREDGRAPH_URL.format(id="DGS10")


class FakeResponse:
    def __init__(self, status_code: int = 200, text: str = DGS10_CSV) -> None:
        self.status_code = status_code
        self.text = text


class FakeGet:
    """Stand-in for requests.get: replays outcomes in order and records each call."""

    def __init__(self, *outcomes: FakeResponse | Exception) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeSleep:
    def __init__(self) -> None:
        self.sleeps: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.sleeps.append(seconds)


def _provider(get: FakeGet, sleep: FakeSleep | None = None) -> FredProvider:
    return FredProvider(get=get, sleep=sleep if sleep is not None else FakeSleep())


def test_unsorted_dates_are_returned_sorted() -> None:
    body = "observation_date,DGS10\n2024-01-03,3.91\n2024-01-02,3.95\n"
    frame = _provider(FakeGet(FakeResponse(text=body))).series(
        ["DGS10"], date(2024, 1, 1), date(2024, 1, 4)
    )
    assert list(frame.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert list(frame["DGS10"]) == [3.95, 3.91]


def test_duplicate_dates_raise() -> None:
    body = "observation_date,DGS10\n2024-01-02,3.95\n2024-01-02,3.91\n"
    with pytest.raises(MacroProviderError, match="duplicate dates"):
        _provider(FakeGet(FakeResponse(text=body))).series(
            ["DGS10"], date(2024, 1, 1), date(2024, 1, 4)
        )


def test_duplicate_ids_are_fetched_once_and_give_one_column() -> None:
    get = FakeGet(FakeResponse())
    frame = _provider(get).series(["DGS10", "DGS10"], date(2024, 1, 1), date(2024, 1, 4))
    assert list(frame.columns) == ["DGS10"]
    assert len(get.calls) == 1


def test_no_ids_give_an_empty_float_frame_on_a_datetime_index() -> None:
    get = FakeGet()
    frame = _provider(get).series([], date(2024, 1, 1), date(2024, 1, 4))
    assert frame.empty
    assert len(frame.columns) == 0
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.index.dtype == "datetime64[ns]"
    assert get.calls == []


def test_a_header_only_csv_is_an_empty_float_column() -> None:
    get = FakeGet(FakeResponse(text="observation_date,DGS10\n"))
    frame = _provider(get).series(["DGS10"], date(2024, 1, 1), date(2024, 1, 4))
    assert list(frame.columns) == ["DGS10"]
    assert len(frame) == 0
    assert frame["DGS10"].dtype == "float64"
    assert frame.index.dtype == "datetime64[ns]"


def test_parses_the_csv_maps_dot_to_nan_and_trims_to_the_window() -> None:
    get = FakeGet(FakeResponse(), FakeResponse())
    provider = _provider(get)

    full = provider.series(["DGS10"], date(2024, 1, 1), date(2024, 1, 4))
    trimmed = provider.series(["DGS10"], date(2024, 1, 2), date(2024, 1, 3))

    assert list(full.columns) == ["DGS10"]
    assert list(full.index) == [pd.Timestamp(f"2024-01-0{day}") for day in (1, 2, 3, 4)]
    assert full.index.dtype == "datetime64[ns]"
    assert full.index.name is None
    assert full.columns.name is None
    assert full["DGS10"].dtype == "float64"
    assert math.isnan(full.at[pd.Timestamp("2024-01-01"), "DGS10"])
    assert full.at[pd.Timestamp("2024-01-04"), "DGS10"] == 3.99
    assert list(trimmed.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]


def test_the_legacy_upper_case_date_header_is_accepted() -> None:
    legacy = DGS10_CSV.replace("observation_date", "DATE")
    get = FakeGet(FakeResponse(text=legacy))

    frame = _provider(get).series(["DGS10"], date(2024, 1, 1), date(2024, 1, 4))

    assert list(frame.columns) == ["DGS10"]
    assert list(frame.index) == [pd.Timestamp(f"2024-01-0{day}") for day in (1, 2, 3, 4)]
    assert math.isnan(frame.at[pd.Timestamp("2024-01-01"), "DGS10"])
    assert frame.at[pd.Timestamp("2024-01-04"), "DGS10"] == 3.99


def test_series_are_columns_in_request_order_on_the_union_of_their_dates() -> None:
    get = FakeGet(FakeResponse(text=VIX_CSV), FakeResponse(text=DGS10_CSV))

    frame = _provider(get).series(["VIXCLS", "DGS10"], date(2024, 1, 2), date(2024, 1, 5))

    assert list(frame.columns) == ["VIXCLS", "DGS10"]
    assert [url for url, _ in get.calls] == [FREDGRAPH_URL.format(id="VIXCLS"), DGS10_URL]
    assert list(frame.index) == [pd.Timestamp(f"2024-01-0{day}") for day in (2, 3, 4, 5)]
    assert math.isnan(frame.at[pd.Timestamp("2024-01-04"), "VIXCLS"])
    assert math.isnan(frame.at[pd.Timestamp("2024-01-05"), "DGS10"])


def test_requests_use_the_library_default_user_agent_and_no_custom_header() -> None:
    get = FakeGet(FakeResponse())
    _provider(get).series(["DGS10"], date(2024, 1, 1), date(2024, 1, 4))
    [(url, kwargs)] = get.calls
    assert url == DGS10_URL
    assert kwargs == {"timeout": REQUEST_TIMEOUT}


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(FakeResponse(status_code=429), id="429"),
        pytest.param(FakeResponse(status_code=500), id="500"),
        pytest.param(FakeResponse(status_code=503), id="503"),
        pytest.param(requests.Timeout("slow"), id="timeout"),
        pytest.param(requests.ConnectionError("reset"), id="connection"),
    ],
)
def test_a_transient_failure_is_retried(failure: FakeResponse | Exception) -> None:
    get, sleep = FakeGet(failure, FakeResponse()), FakeSleep()
    frame = _provider(get, sleep).series(["DGS10"], date(2024, 1, 2), date(2024, 1, 4))
    assert len(frame) == 3
    assert len(get.calls) == 2
    assert sleep.sleeps == [1.0]


def test_the_backoff_doubles_and_exhaustion_raises() -> None:
    get, sleep = FakeGet(*[FakeResponse(status_code=503)] * (MAX_RETRIES + 1)), FakeSleep()
    with pytest.raises(MacroProviderError, match=r"HTTP 503; gave up after 3 retries"):
        _provider(get, sleep).series(["DGS10"], date(2024, 1, 2), date(2024, 1, 4))
    assert len(get.calls) == MAX_RETRIES + 1
    assert sleep.sleeps == [1.0, 2.0, 4.0]


def test_a_non_429_4xx_is_not_retried() -> None:
    get, sleep = FakeGet(FakeResponse(status_code=404)), FakeSleep()
    with pytest.raises(MacroProviderError, match="HTTP 404"):
        _provider(get, sleep).series(["DGS10"], date(2024, 1, 2), date(2024, 1, 4))
    assert len(get.calls) == 1
    assert sleep.sleeps == []


def test_another_request_error_is_not_retried() -> None:
    get = FakeGet(requests.exceptions.InvalidURL("bad url"))
    with pytest.raises(MacroProviderError, match="bad url"):
        _provider(get).series(["DGS10"], date(2024, 1, 2), date(2024, 1, 4))
    assert len(get.calls) == 1


@pytest.mark.parametrize(
    ("body", "message"),
    [
        pytest.param(
            "<!DOCTYPE html><html><body>Maintenance</body></html>",
            "not a fredgraph CSV",
            id="html-with-200",
        ),
        pytest.param("", "empty body", id="empty"),
        pytest.param("  \n", "empty body", id="blank"),
        pytest.param("2024-01-02,3.95\n", "not a fredgraph CSV", id="missing-header"),
        pytest.param(
            "date,value\n2024-01-02,3.95\n", "not a fredgraph CSV", id="unexpected-header"
        ),
        pytest.param(
            "observation_date,DGS10\n2024-01-02,n/a\n", "malformed CSV", id="non-numeric-value"
        ),
        pytest.param(
            "observation_date,VIXCLS\n2024-01-02,13.2\n", "not 'DGS10'", id="another-series-id"
        ),
    ],
)
def test_a_bad_body_raises(body: str, message: str) -> None:
    get = FakeGet(FakeResponse(text=body))
    with pytest.raises(MacroProviderError, match=message):
        _provider(get).series(["DGS10"], date(2024, 1, 2), date(2024, 1, 4))
