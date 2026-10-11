"""FundamentalsProvider backed by SEC EDGAR companyfacts (data.sec.gov XBRL API)."""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd
import requests

from hedge_fund.engine.fundamentals.metrics import CURATED_TAGS, REPORT_FORMS
from hedge_fund.ports.fundamentals import FACT_COLUMNS, FundamentalsProviderError, coerce_facts

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
USER_AGENT_ENV = "SEC_USER_AGENT"
MIN_REQUEST_INTERVAL = 0.12  # seconds; SEC allows at most 10 requests per second
MAX_RETRIES = 3
BACKOFF_SECONDS = 1.0  # 1 s, 2 s, 4 s
REQUEST_TIMEOUT = 30
UNITS: frozenset[str] = frozenset({"USD", "shares"})

_CIK = re.compile(r"\d{10}", re.ASCII)

HttpGet = Callable[..., Any]


class EdgarProvider:
    """Raw companyfacts for tickers with a CIK, kept to curated tags, units and report forms.

    SEC requires a User-Agent with contact details (argument or SEC_USER_AGENT). Requests
    are spaced MIN_REQUEST_INTERVAL apart. HTTP 429, 5xx, timeouts and connection errors are
    retried MAX_RETRIES times with exponential backoff; HTTP 404 means no rows.
    """

    def __init__(
        self,
        ciks: Mapping[str, str],
        *,
        get: HttpGet | None = None,
        user_agent: str | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        agent = user_agent if user_agent is not None else os.environ.get(USER_AGENT_ENV)
        if not agent:
            raise FundamentalsProviderError(
                f"{USER_AGENT_ENV} is not set; SEC requires a User-Agent like 'Name email@host'"
            )
        for ticker, cik in ciks.items():
            if not _CIK.fullmatch(cik):
                raise FundamentalsProviderError(
                    f"invalid CIK for {ticker!r}: {cik[:20]!r} (expected 10 digits)"
                )
        self._ciks = dict(ciks)
        self._headers = {"User-Agent": agent}
        self._get: HttpGet = get if get is not None else requests.get
        self._clock = clock if clock is not None else time.monotonic
        self._sleep = sleep if sleep is not None else time.sleep
        self._last_request: float | None = None

    def facts(self, tickers: Sequence[str]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for ticker in dict.fromkeys(tickers):
            cik = self._ciks.get(ticker)
            payload = self._fetch(cik) if cik is not None else None
            if payload is None:
                continue
            try:
                rows.extend(_payload_rows(ticker, payload))
            except (KeyError, TypeError, AttributeError) as exc:
                raise FundamentalsProviderError(
                    f"malformed companyfacts for {ticker}: {exc!r}"
                ) from exc
        try:
            return coerce_facts(pd.DataFrame(rows, columns=list(FACT_COLUMNS)))
        except (ValueError, TypeError) as exc:
            raise FundamentalsProviderError(f"malformed companyfacts values: {exc!r}") from exc

    def _fetch(self, cik: str) -> Mapping[str, Any] | None:
        """Parsed companyfacts JSON for `cik`, or None on HTTP 404."""
        url = COMPANYFACTS_URL.format(cik=cik)
        failure = ""
        for attempt in range(MAX_RETRIES + 1):
            if attempt:
                self._sleep(BACKOFF_SECONDS * 2 ** (attempt - 1))
            self._throttle()
            try:
                response = self._get(url, headers=self._headers, timeout=REQUEST_TIMEOUT)
            except (requests.Timeout, requests.ConnectionError) as exc:
                failure = f"{type(exc).__name__}: {exc}"
                continue
            except requests.RequestException as exc:
                raise FundamentalsProviderError(f"{url}: {exc}") from exc
            status = response.status_code
            if status == 404:
                return None
            if status == 429 or status >= 500:
                failure = f"HTTP {status}"
                continue
            if status != 200:
                raise FundamentalsProviderError(f"{url}: HTTP {status}")
            return _json(response, url)
        raise FundamentalsProviderError(f"{url}: {failure}; gave up after {MAX_RETRIES} retries")

    def _throttle(self) -> None:
        if self._last_request is not None:
            wait = self._last_request + MIN_REQUEST_INTERVAL - self._clock()
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()


def _json(response: Any, url: str) -> Mapping[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise FundamentalsProviderError(f"{url}: invalid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise FundamentalsProviderError(f"{url}: JSON payload is not an object")
    return payload


def _payload_rows(ticker: str, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Curated-tag facts in USD or shares units from report forms, as FACT_COLUMNS rows.

    companyfacts' fy, fp and frame fields are dropped.
    """
    taxonomies = payload.get("facts") or {}
    rows: list[dict[str, Any]] = []
    for tag in CURATED_TAGS:
        taxonomy, _, concept = tag.partition(":")
        units = ((taxonomies.get(taxonomy) or {}).get(concept) or {}).get("units") or {}
        for unit, entries in units.items():
            if unit in UNITS:
                if not isinstance(entries, list):
                    raise FundamentalsProviderError(
                        f"{ticker} {tag} {unit}: unit entries are not a list"
                    )
                rows.extend(
                    _row(ticker, tag, entry)
                    for entry in entries
                    if entry.get("form") in REPORT_FORMS
                )
    return rows


def _row(ticker: str, tag: str, entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "tag": tag,
        "start": entry.get("start"),
        "end": entry["end"],
        "value": entry["val"],
        "filed": entry["filed"],
        "form": entry["form"],
        "accn": entry["accn"],
    }
