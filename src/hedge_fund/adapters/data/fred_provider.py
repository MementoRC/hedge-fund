"""MacroProvider backed by FRED's keyless fredgraph CSV (fred.stlouisfed.org).

Only never-revised, market-priced daily series are read, so no ALFRED vintages and no API key
are needed. FRED dates an observation at its period start; for these daily series that is the
observation day. Requests carry the library-default User-Agent and no custom header, because
fred.stlouisfed.org stalls on custom ones.
"""

from __future__ import annotations

import io
import time
from collections.abc import Callable, Sequence
from datetime import date
from typing import Any

import pandas as pd
import requests

from hedge_fund.ports.macro import MacroProviderError

FREDGRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
DATE_HEADERS = ("observation_date", "DATE")  # FRED's current header and its legacy one
MISSING_VALUE = "."
MAX_RETRIES = 3
BACKOFF_SECONDS = 1.0  # 1 s, 2 s, 4 s
REQUEST_TIMEOUT = 30

HttpGet = Callable[..., Any]


class FredProvider:
    """FRED series as float columns by observation date; FRED's "." (no value) is NaN.

    HTTP 429, 5xx, timeouts and connection errors are retried MAX_RETRIES times with
    exponential backoff. Every vendor or parse failure raises MacroProviderError, including a
    body that is not a fredgraph CSV but came with HTTP 200 (FRED's HTML maintenance page).
    """

    def __init__(
        self,
        *,
        get: HttpGet | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._get: HttpGet = get if get is not None else requests.get
        self._sleep = sleep if sleep is not None else time.sleep

    def series(self, ids: Sequence[str], start: date, end: date) -> pd.DataFrame:
        unique = list(dict.fromkeys(ids))
        if not unique:
            return pd.DataFrame(index=pd.DatetimeIndex([], dtype="datetime64[ns]"), dtype="float64")
        found = {series_id: self._series(series_id) for series_id in unique}
        frame = pd.DataFrame(found, dtype="float64").sort_index()
        index = pd.DatetimeIndex(frame.index)
        rows = (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
        return frame.loc[rows, unique].rename_axis(index=None, columns=None)

    def _series(self, series_id: str) -> pd.Series:
        url = FREDGRAPH_URL.format(id=series_id)
        return _parse(self._fetch(url), series_id, url)

    def _fetch(self, url: str) -> str:
        """The body at ``url``, retrying transient failures."""
        failure = ""
        for attempt in range(MAX_RETRIES + 1):
            if attempt:
                self._sleep(BACKOFF_SECONDS * 2 ** (attempt - 1))
            try:
                response = self._get(url, timeout=REQUEST_TIMEOUT)
            except (requests.Timeout, requests.ConnectionError) as exc:
                failure = f"{type(exc).__name__}: {exc}"
                continue
            except requests.RequestException as exc:
                raise MacroProviderError(f"{url}: {exc}") from exc
            status = response.status_code
            if status == 429 or status >= 500:
                failure = f"HTTP {status}"
                continue
            if status != 200:
                raise MacroProviderError(f"{url}: HTTP {status}")
            return str(response.text)
        raise MacroProviderError(f"{url}: {failure}; gave up after {MAX_RETRIES} retries")


def _parse(text: str, series_id: str, url: str) -> pd.Series:
    """The ``series_id`` values of a fredgraph CSV body, by observation date (ns, unnamed)."""
    lines = text.strip().splitlines()
    if not lines:
        raise MacroProviderError(f"{url}: empty body")
    header = [cell.strip() for cell in lines[0].split(",")]
    if len(header) != 2 or header[0] not in DATE_HEADERS:
        raise MacroProviderError(f"{url}: not a fredgraph CSV, header {lines[0][:80]!r}")
    if header[1] != series_id:
        raise MacroProviderError(f"{url}: the CSV is for {header[1]!r}, not {series_id!r}")
    try:
        frame = pd.read_csv(io.StringIO(text), na_values=[MISSING_VALUE], keep_default_na=False)
        dates = pd.DatetimeIndex(pd.to_datetime(frame[header[0]], format="%Y-%m-%d"))
        values = pd.to_numeric(frame[series_id]).to_numpy(dtype="float64")
    except (ValueError, TypeError) as exc:
        raise MacroProviderError(f"{url}: malformed CSV: {exc}") from exc
    if not dates.is_unique:
        raise MacroProviderError(f"{url}: duplicate dates")
    return pd.Series(values, index=dates.as_unit("ns").rename(None), name=series_id).sort_index()
