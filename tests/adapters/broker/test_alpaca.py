"""Tests for the Alpaca paper broker (offline: get, post, clock and sleep injected)."""

from __future__ import annotations

import inspect
import time
from typing import Any

import pytest
import requests

from hedge_fund.adapters.broker.alpaca import BASE_URL, AlpacaPaperBroker
from hedge_fund.domain.execution import Account
from hedge_fund.domain.orders import Holding
from hedge_fund.ports.broker import BrokerError

KEY_ID = "test-key-id"
SECRET = "test-secret"  # noqa: S105  # nosec B105 (fake test key)
HEADERS = {"APCA-API-KEY-ID": KEY_ID, "APCA-API-SECRET-KEY": SECRET}
EQUITY = {"equity": "1000.5", "cash": "12.0"}


class FakeTime:
    """Monotonic clock that advances only when slept on or when a fake request takes time."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeResponse:
    def __init__(self, status_code: int = 200, payload: Any = None) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeGet:
    """Stand-in for requests.get: replays outcomes in order and records each call."""

    def __init__(
        self,
        *outcomes: FakeResponse | Exception,
        clock: FakeTime | None = None,
        latency: float = 0.0,
    ) -> None:
        self.outcomes = list(outcomes)
        self.clock = clock
        self.latency = latency
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, kwargs))
        if self.clock is not None:
            self.clock.now += self.latency
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakePost(FakeGet):
    """Stand-in for requests.post: replays outcomes in order and records each call."""


def _broker(
    get: FakeGet | None = None, post: FakePost | None = None, clock: FakeTime | None = None
) -> AlpacaPaperBroker:
    fake_time = clock if clock is not None else FakeTime()
    return AlpacaPaperBroker(
        KEY_ID,
        SECRET,
        get=get if get is not None else FakeGet(),
        post=post if post is not None else FakePost(),
        clock=fake_time.clock,
        sleep=fake_time.sleep,
    )


# --- construction -----------------------------------------------------------------------


def test_base_url_is_the_paper_host() -> None:
    assert BASE_URL == "https://paper-api.alpaca.markets"


def test_no_host_override() -> None:
    parameters = inspect.signature(AlpacaPaperBroker).parameters
    assert list(parameters) == ["key_id", "secret_key", "get", "post", "clock", "sleep"]


def test_defaults_are_requests_and_monotonic_time() -> None:
    broker = AlpacaPaperBroker(KEY_ID, SECRET)
    assert broker._get is requests.get
    assert broker._post is requests.post
    assert broker._clock is time.monotonic
    assert broker._sleep is time.sleep


@pytest.mark.parametrize(
    ("key_id", "secret_key", "missing"),
    [
        (None, SECRET, "APCA_API_KEY_ID"),
        ("", SECRET, "APCA_API_KEY_ID"),
        (KEY_ID, None, "APCA_API_SECRET_KEY"),
        (KEY_ID, "", "APCA_API_SECRET_KEY"),
    ],
)
def test_missing_key_raises_at_construction(
    monkeypatch: pytest.MonkeyPatch, key_id: str | None, secret_key: str | None, missing: str
) -> None:
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(BrokerError, match=f"{missing} is not set") as info:
        AlpacaPaperBroker(key_id, secret_key)
    assert KEY_ID not in str(info.value)
    assert SECRET not in str(info.value)


def test_keys_fall_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APCA_API_KEY_ID", "env-id")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "env-secret")
    get = FakeGet(FakeResponse(200, EQUITY))
    AlpacaPaperBroker(get=get, post=FakePost()).account()
    assert get.calls[0][1]["headers"] == {
        "APCA-API-KEY-ID": "env-id",
        "APCA-API-SECRET-KEY": "env-secret",
    }


def test_arguments_take_precedence_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APCA_API_KEY_ID", "env-id")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "env-secret")
    get = FakeGet(FakeResponse(200, EQUITY))
    _broker(get).account()
    assert get.calls[0][1]["headers"] == HEADERS


# --- account ----------------------------------------------------------------------------


def test_account_request_and_equity() -> None:
    get = FakeGet(FakeResponse(200, EQUITY))
    assert _broker(get).account() == Account(equity=1000.5)
    assert get.calls == [
        (f"{BASE_URL}/v2/account", {"headers": HEADERS, "params": None, "timeout": 30})
    ]


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(ValueError("not json"), id="invalid-json"),
        pytest.param(["1000.5"], id="not-an-object"),
        pytest.param({"cash": "1"}, id="missing"),
        pytest.param({"equity": 1000.5}, id="not-a-string"),
        pytest.param({"equity": "abc"}, id="not-a-number"),
        pytest.param({"equity": "nan"}, id="nan"),
        pytest.param({"equity": "inf"}, id="inf"),
    ],
)
def test_account_rejects_a_malformed_payload(payload: Any) -> None:
    with pytest.raises(BrokerError, match="GET /v2/account"):
        _broker(FakeGet(FakeResponse(200, payload))).account()


def test_invalid_json_is_chained() -> None:
    bad = ValueError("not json")
    with pytest.raises(BrokerError, match="invalid JSON") as info:
        _broker(FakeGet(FakeResponse(200, bad))).account()
    assert info.value.__cause__ is bad


@pytest.mark.parametrize("status", [201, 400, 401, 403, 404])
def test_account_non_200_raises_without_retry(status: int) -> None:
    get = FakeGet(FakeResponse(status, {"message": "nope"}))
    with pytest.raises(BrokerError, match=f"GET /v2/account: HTTP {status}"):
        _broker(get).account()
    assert len(get.calls) == 1


# --- retry and throttle -----------------------------------------------------------------


def test_retries_429_5xx_timeouts_and_connection_errors_with_backoff() -> None:
    fake_time = FakeTime()
    get = FakeGet(
        FakeResponse(429),
        requests.ConnectionError("reset"),
        requests.Timeout("slow"),
        FakeResponse(200, EQUITY),
    )
    assert _broker(get, clock=fake_time).account() == Account(equity=1000.5)
    assert len(get.calls) == 4
    assert fake_time.sleeps == [1.0, 2.0, 4.0]


@pytest.mark.parametrize("status", [429, 500, 503])
def test_gives_up_after_three_retries_on_status(status: int) -> None:
    get = FakeGet(*[FakeResponse(status)] * 4)
    with pytest.raises(
        BrokerError, match=f"GET /v2/account: HTTP {status}; gave up after 3 retries"
    ):
        _broker(get).account()
    assert len(get.calls) == 4


@pytest.mark.parametrize(
    "error", [requests.Timeout("slow"), requests.ConnectionError("down")], ids=["timeout", "conn"]
)
def test_gives_up_after_three_retries_on_transport_error(error: Exception) -> None:
    get = FakeGet(*[error] * 4)
    with pytest.raises(BrokerError, match="gave up after 3 retries") as info:
        _broker(get).account()
    assert info.value.__cause__ is error
    assert len(get.calls) == 4


def test_other_request_errors_fail_without_retry() -> None:
    boom = requests.exceptions.InvalidURL("bad url")
    get = FakeGet(boom)
    with pytest.raises(BrokerError, match="GET /v2/account") as info:
        _broker(get).account()
    assert info.value.__cause__ is boom
    assert len(get.calls) == 1


def test_requests_are_spaced_at_least_0_3_seconds() -> None:
    fake_time = FakeTime()
    get = FakeGet(FakeResponse(200, EQUITY), FakeResponse(200, EQUITY))
    broker = _broker(get, clock=fake_time)
    broker.account()
    broker.account()
    assert fake_time.sleeps == [pytest.approx(0.3)]


def test_no_wait_when_enough_time_has_passed() -> None:
    fake_time = FakeTime()
    get = FakeGet(
        FakeResponse(200, EQUITY), FakeResponse(200, EQUITY), clock=fake_time, latency=0.5
    )
    broker = _broker(get, clock=fake_time)
    broker.account()
    broker.account()
    assert fake_time.sleeps == []


# --- positions --------------------------------------------------------------------------


def _position(**overrides: Any) -> dict[str, Any]:
    return {
        "symbol": "AAPL",
        "qty": "10",
        "side": "long",
        "asset_class": "us_equity",
        "avg_entry_price": "150.0",
        **overrides,
    }


def _without(key: str) -> dict[str, Any]:
    entry = _position()
    del entry[key]
    return entry


def _holdings(payload: Any) -> tuple[Holding, ...]:
    return _broker(FakeGet(FakeResponse(200, payload))).positions()


def test_positions_request() -> None:
    get = FakeGet(FakeResponse(200, []))
    assert _broker(get).positions() == ()
    assert get.calls == [
        (f"{BASE_URL}/v2/positions", {"headers": HEADERS, "params": None, "timeout": 30})
    ]


def test_positions_are_signed_by_side_and_sorted() -> None:
    payload = [
        _position(symbol="MSFT", qty="10"),
        _position(symbol="AAPL", qty="-5", side="short"),
        _position(symbol="TSLA", qty="7", side="short"),  # positive qty, short side
        _position(symbol="KO", qty="-3", side="long"),  # negative qty, long side
    ]
    assert _holdings(payload) == (
        Holding(ticker="AAPL", shares=-5),
        Holding(ticker="KO", shares=3),
        Holding(ticker="MSFT", shares=10),
        Holding(ticker="TSLA", shares=-7),
    )


@pytest.mark.parametrize(("qty", "shares"), [("100", 100), ("100.0", 100), ("-100", 100)])
def test_whole_number_qty_forms(qty: str, shares: int) -> None:
    assert _holdings([_position(qty=qty)]) == (Holding(ticker="AAPL", shares=shares),)


@pytest.mark.parametrize("qty", ["0", "0.0", "-0"])
def test_zero_share_positions_are_dropped(qty: str) -> None:
    payload = [_position(qty=qty), _position(symbol="MSFT")]
    assert _holdings(payload) == (Holding(ticker="MSFT", shares=10),)


@pytest.mark.parametrize(
    "qty", ["1.5", "-0.25", "nan", "NaN", "inf", "-Infinity", "1e2", "1E2", "abc", ""]
)
def test_rejects_qty_that_is_not_a_whole_number(qty: str) -> None:
    with pytest.raises(BrokerError, match=r"GET /v2/positions: AAPL qty"):
        _holdings([_position(qty=qty)])


def test_unparseable_qty_is_chained() -> None:
    with pytest.raises(BrokerError, match="is not a number") as info:
        _holdings([_position(qty="abc")])
    assert info.value.__cause__ is not None


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(ValueError("not json"), id="invalid-json"),
        pytest.param({"symbol": "AAPL"}, id="not-an-array"),
        pytest.param(["AAPL"], id="entry-not-an-object"),
        pytest.param([_without("symbol")], id="missing-symbol"),
        pytest.param([_position(symbol=5)], id="non-string-symbol"),
        pytest.param([_position(symbol="")], id="empty-symbol"),
        pytest.param([_without("qty")], id="missing-qty"),
        pytest.param([_position(qty=10)], id="non-string-qty"),
        pytest.param([_without("side")], id="missing-side"),
        pytest.param([_position(side="flat")], id="unknown-side"),
        pytest.param([_position(side=["long"])], id="unhashable-side"),
        pytest.param([_without("asset_class")], id="missing-asset-class"),
        pytest.param([_position(asset_class="crypto")], id="non-equity"),
    ],
)
def test_rejects_malformed_positions(payload: Any) -> None:
    with pytest.raises(BrokerError, match="GET /v2/positions"):
        _holdings(payload)


def test_rejects_duplicate_symbol() -> None:
    with pytest.raises(BrokerError, match="duplicate symbol 'AAPL'"):
        _holdings([_position(), _position(qty="5")])


def test_positions_non_200_raises() -> None:
    with pytest.raises(BrokerError, match="GET /v2/positions: HTTP 401"):
        _broker(FakeGet(FakeResponse(401))).positions()
