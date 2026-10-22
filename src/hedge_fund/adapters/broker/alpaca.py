"""Broker backed by Alpaca's paper-trading REST API.

The host is the paper endpoint, fixed in BASE_URL with no constructor or env override, so
no live-money path exists. Requests are spaced MIN_REQUEST_INTERVAL apart; HTTP 429, 5xx,
timeouts and connection errors are retried MAX_RETRIES times with exponential backoff.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from typing import Any

import requests

from hedge_fund.domain.execution import Account
from hedge_fund.domain.orders import Holding
from hedge_fund.ports.broker import BrokerError

BASE_URL = "https://paper-api.alpaca.markets"
KEY_ID_ENV = "APCA_API_KEY_ID"
SECRET_KEY_ENV = "APCA_API_SECRET_KEY"  # noqa: S105  # nosec B105 (env var name, not a secret)
MIN_REQUEST_INTERVAL = 0.3  # seconds; stays under Alpaca's 200 requests per minute
MAX_RETRIES = 3
BACKOFF_SECONDS = 1.0  # 1 s, 2 s, 4 s
REQUEST_TIMEOUT = 30

ACCOUNT_PATH = "/v2/account"
POSITIONS_PATH = "/v2/positions"

_SIDE_SIGN = {"long": 1, "short": -1}

HttpCall = Callable[..., Any]


class AlpacaPaperBroker:
    """Alpaca paper account: equity, signed holdings and market-on-open order submission.

    Keys come from the arguments or APCA_API_KEY_ID / APCA_API_SECRET_KEY and are sent as
    the APCA-API-KEY-ID / APCA-API-SECRET-KEY headers. Every failure raises BrokerError.
    """

    def __init__(
        self,
        key_id: str | None = None,
        secret_key: str | None = None,
        *,
        get: HttpCall | None = None,
        post: HttpCall | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        key_id = key_id if key_id is not None else os.environ.get(KEY_ID_ENV)
        secret_key = secret_key if secret_key is not None else os.environ.get(SECRET_KEY_ENV)
        if not key_id:
            raise BrokerError(f"{KEY_ID_ENV} is not set")
        if not secret_key:
            raise BrokerError(f"{SECRET_KEY_ENV} is not set")
        self._headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret_key}
        self._get: HttpCall = get if get is not None else requests.get
        self._post: HttpCall = post if post is not None else requests.post
        self._clock = clock if clock is not None else time.monotonic
        self._sleep = sleep if sleep is not None else time.sleep
        self._last_request: float | None = None

    def account(self) -> Account:
        """Account equity; the body must hold a string ``equity`` parsing to a finite float."""
        what = f"GET {ACCOUNT_PATH}"
        body = self._get_json(ACCOUNT_PATH)
        equity = body.get("equity") if isinstance(body, dict) else None
        if not isinstance(equity, str):
            raise BrokerError(f"{what}: equity is missing or not a string")
        try:
            value = float(equity)
        except ValueError as exc:
            raise BrokerError(f"{what}: equity {equity!r} is not a number") from exc
        if not math.isfinite(value):
            raise BrokerError(f"{what}: equity {equity!r} is not finite")
        return Account(equity=value)

    def positions(self) -> tuple[Holding, ...]:
        """Signed whole-share holdings sorted by ticker, one per ticker, none with zero shares.

        The sign comes from ``side`` (Alpaca does not say whether a short ``qty`` is
        negative). Raises BrokerError on a malformed entry, a non-equity asset, a ``qty`` that
        is not a whole number, or a duplicate symbol.
        """
        body = self._get_json(POSITIONS_PATH)
        if not isinstance(body, list):
            raise BrokerError(f"GET {POSITIONS_PATH}: body is not an array")
        shares: dict[str, int] = {}
        for entry in body:
            ticker, signed = _position(entry)
            if ticker in shares:
                raise BrokerError(f"GET {POSITIONS_PATH}: duplicate symbol {ticker!r}")
            shares[ticker] = signed
        return tuple(
            Holding(ticker=ticker, shares=signed)
            for ticker, signed in sorted(shares.items())
            if signed != 0
        )

    def _get_json(self, path: str) -> Any:
        """Parsed body of a GET that must answer 200."""
        response = self._get_response(path)
        if response.status_code != 200:
            raise BrokerError(f"GET {path}: HTTP {response.status_code}")
        return _json(response, f"GET {path}")

    def _get_response(self, path: str, params: dict[str, str] | None = None) -> Any:
        return self._call(
            f"GET {path}",
            lambda: self._get(
                BASE_URL + path, headers=self._headers, params=params, timeout=REQUEST_TIMEOUT
            ),
        )

    def _call(self, what: str, send: Callable[[], Any]) -> Any:
        """Response of ``send()`` after throttling, with retries and backoff.

        429, 5xx, timeouts and connection errors are retried MAX_RETRIES times, then raise
        BrokerError (chained to the last transport error, if any). Other request errors
        raise at once. Every other status is returned for the caller to map.
        """
        failure = ""
        cause: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            if attempt:
                self._sleep(BACKOFF_SECONDS * 2 ** (attempt - 1))
            self._throttle()
            try:
                response = send()
            except (requests.Timeout, requests.ConnectionError) as exc:
                failure, cause = f"{type(exc).__name__}: {exc}", exc
                continue
            except requests.RequestException as exc:
                raise BrokerError(f"{what}: {exc}") from exc
            status = response.status_code
            if status == 429 or status >= 500:
                failure, cause = f"HTTP {status}", None
                continue
            return response
        raise BrokerError(f"{what}: {failure}; gave up after {MAX_RETRIES} retries") from cause

    def _throttle(self) -> None:
        if self._last_request is not None:
            wait = self._last_request + MIN_REQUEST_INTERVAL - self._clock()
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()


def _json(response: Any, what: str) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise BrokerError(f"{what}: invalid JSON: {exc}") from exc


def _position(entry: Any) -> tuple[str, int]:
    """(ticker, signed shares) of one /v2/positions entry."""
    what = f"GET {POSITIONS_PATH}"
    if not isinstance(entry, dict):
        raise BrokerError(f"{what}: position is not an object")
    symbol, qty, side = entry.get("symbol"), entry.get("qty"), entry.get("side")
    if not isinstance(symbol, str) or not symbol:
        raise BrokerError(f"{what}: position symbol is missing or not a string")
    if not isinstance(qty, str) or not isinstance(side, str) or side not in _SIDE_SIGN:
        raise BrokerError(f"{what}: {symbol} has a malformed qty or side")
    asset_class = entry.get("asset_class")
    if asset_class != "us_equity":
        raise BrokerError(f"{what}: {symbol} asset_class {asset_class!r} is not us_equity")
    return symbol, _SIDE_SIGN[side] * abs(_whole_shares(qty, symbol))


def _whole_shares(qty: str, symbol: str) -> int:
    """``qty`` as an int: finite, integral and written without an exponent."""
    what = f"GET {POSITIONS_PATH}"
    try:
        value = Decimal(qty)
    except InvalidOperation as exc:
        raise BrokerError(f"{what}: {symbol} qty {qty!r} is not a number") from exc
    if not value.is_finite() or "e" in qty.lower() or value != value.to_integral_value():
        raise BrokerError(f"{what}: {symbol} qty {qty!r} is not a whole number")
    return int(value)
