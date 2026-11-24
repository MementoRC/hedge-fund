"""MacroProvider backed by FRED's keyless fredgraph CSV (fred.stlouisfed.org).

Only never-revised, market-priced daily series are read, so no ALFRED vintages and no API key
are needed. FRED dates an observation at its period start; for these daily series that is the
observation day. Requests carry the library-default User-Agent and no custom header, because
fred.stlouisfed.org stalls on custom ones.
"""

from __future__ import annotations

import io
import threading
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
REQUEST_TIMEOUT = 30  # per socket read only: a trickling server resets it with every byte
TOTAL_DEADLINE = 60.0  # wall-clock cap per series() call, over every id, retry and backoff sleep
CHUNK_SIZE = 8192
WORKER_NAME = "fred-series"

# requests.get-compatible: called as get(url, timeout=..., stream=True); the response must offer
# status_code, iter_content(chunk_size=...) yielding bytes, and close().
HttpGet = Callable[..., Any]


class FredProvider:
    """FRED series as float columns by observation date; FRED's "." (no value) is NaN.

    HTTP 429, 5xx, timeouts and connection errors are retried MAX_RETRIES times with
    exponential backoff. One ``deadline`` (seconds, default TOTAL_DEADLINE) bounds a whole
    series() call, bodies read in chunks and backoff sleeps included; expiry raises
    MacroProviderError and is not retried. The work runs in a worker thread and the caller waits
    at most ``deadline`` seconds for it, so series() returns or raises within the deadline
    (plus scheduling) even when a read, DNS lookup or parse blocks. An abandoned worker is a
    daemon; its chunk checks and request timeouts (capped at the time left) end it within one
    more read. Every
    vendor or parse failure raises MacroProviderError, including a body that is not a fredgraph
    CSV but came with HTTP 200 (FRED's HTML maintenance page).
    """

    def __init__(
        self,
        *,
        get: HttpGet | None = None,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        deadline: float = TOTAL_DEADLINE,
    ) -> None:
        self._get: HttpGet = get if get is not None else requests.get
        self._sleep = sleep if sleep is not None else time.sleep
        self._clock = clock
        self._deadline = deadline

    def series(self, ids: Sequence[str], start: date, end: date) -> pd.DataFrame:
        unique = list(dict.fromkeys(ids))
        if not unique:
            return pd.DataFrame(index=pd.DatetimeIndex([], dtype="datetime64[ns]"), dtype="float64")
        expires = self._clock() + self._deadline
        found = self._within_deadline(unique, expires)
        frame = pd.DataFrame(found, dtype="float64").sort_index()
        index = pd.DatetimeIndex(frame.index)
        rows = (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
        return frame.loc[rows, unique].rename_axis(index=None, columns=None)

    def _within_deadline(self, ids: list[str], expires: float) -> dict[str, pd.Series]:
        """Fetch and parse every id in a worker thread, giving up when the deadline passes.

        The watchdog is what makes the deadline hard: a single ``get`` or ``iter_content`` step may
        perform many socket reads (each resetting the per-read timeout) and DNS has no timeout.
        An abandoned worker is a daemon, and its own deadline checks and capped read timeouts end
        it within one further read.
        """
        done = threading.Event()
        outcome: dict[str, Any] = {}

        def work() -> None:
            try:
                outcome["found"] = {sid: self._series(sid, expires) for sid in ids}
            except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
                outcome["error"] = exc
            finally:
                done.set()

        threading.Thread(target=work, name=WORKER_NAME, daemon=True).start()
        if not done.wait(self._deadline):
            names = ", ".join(FREDGRAPH_URL.format(id=sid) for sid in ids)
            raise MacroProviderError(f"{names}: exceeded the {self._deadline:g}s total deadline")
        if "error" in outcome:
            raise outcome["error"]
        found: dict[str, pd.Series] = outcome["found"]
        return found

    def _series(self, series_id: str, expires: float) -> pd.Series:
        url = FREDGRAPH_URL.format(id=series_id)
        return _parse(self._fetch(url, expires), series_id, url)

    def _remaining(self, url: str, expires: float) -> float:
        """Seconds left before ``expires``; raises once none are left."""
        remaining = expires - self._clock()
        if remaining <= 0:
            raise MacroProviderError(f"{url}: exceeded the {self._deadline:g}s total deadline")
        return remaining

    def _fetch(self, url: str, expires: float) -> str:
        """The body at ``url``, retrying transient failures until ``expires``."""
        failure = ""
        for attempt in range(MAX_RETRIES + 1):
            remaining = self._remaining(url, expires)
            if attempt:
                self._sleep(min(BACKOFF_SECONDS * 2 ** (attempt - 1), remaining))
                remaining = self._remaining(url, expires)
            try:
                response = self._get(url, timeout=min(REQUEST_TIMEOUT, remaining), stream=True)
                try:
                    status = response.status_code
                    if status == 429 or status >= 500:
                        failure = f"HTTP {status}"
                        continue
                    if status != 200:
                        raise MacroProviderError(f"{url}: HTTP {status}")
                    return self._read(response, url, expires)
                finally:
                    response.close()
            except (requests.Timeout, requests.ConnectionError) as exc:
                failure = f"{type(exc).__name__}: {exc}"
            except requests.RequestException as exc:
                raise MacroProviderError(f"{url}: {exc}") from exc
        raise MacroProviderError(f"{url}: {failure}; gave up after {MAX_RETRIES} retries")

    def _read(self, response: Any, url: str, expires: float) -> str:
        """The UTF-8 body of a streamed ``response``, checking ``expires`` after every chunk."""
        chunks: list[bytes] = []
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            chunks.append(chunk)
            self._remaining(url, expires)
        try:
            return b"".join(chunks).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MacroProviderError(f"{url}: body is not valid UTF-8: {exc}") from exc


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
