"""Tests for the Alpaca paper broker (offline: get, post, clock and sleep injected)."""

from __future__ import annotations

import dataclasses
import inspect
import time
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
import requests

from hedge_fund.adapters.broker.alpaca import BASE_URL, AlpacaPaperBroker
from hedge_fund.domain.execution import Account, OrderRequest, SubmitResult, SubmitStatus
from hedge_fund.domain.ledger import OrderState, OrderStatus
from hedge_fund.domain.orders import Holding, Order, OrderAction
from hedge_fund.ports.broker import Broker, BrokerError, OrderDataError

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


def test_alpaca_paper_broker_satisfies_the_port() -> None:
    broker: Broker = _broker()
    assert isinstance(broker, AlpacaPaperBroker)


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


# --- submit -----------------------------------------------------------------------------

CID = "hf-2026-10-21-AAPL-sell_short-625"
ORDER = Order(
    ticker="AAPL", action=OrderAction.SELL_SHORT, shares=625, ref_price=100.0, notional=62_500.0
)
REQUEST = OrderRequest(order=ORDER, client_order_id=CID)
ORDERS_URL = f"{BASE_URL}/v2/orders"
LOOKUP_URL = f"{BASE_URL}/v2/orders:by_client_order_id"
NOT_UNIQUE = {"code": 42210000, "message": "client_order_id must be unique"}


def _result(
    status: SubmitStatus, broker_order_id: str | None = None, reason: str | None = None
) -> SubmitResult:
    return SubmitResult(
        request=REQUEST, status=status, broker_order_id=broker_order_id, reason=reason
    )


def _existing(**overrides: Any) -> dict[str, Any]:
    return {
        "id": "ord-9",
        "client_order_id": CID,
        "status": "accepted",
        "symbol": "AAPL",
        **overrides,
    }


def test_submit_posts_a_market_on_open_order() -> None:
    get, post = FakeGet(), FakePost(FakeResponse(200, {"id": "ord-1", "status": "accepted"}))
    assert _broker(get, post).submit(REQUEST) == _result(
        SubmitStatus.ACCEPTED, broker_order_id="ord-1"
    )
    assert post.calls == [
        (
            ORDERS_URL,
            {
                "headers": HEADERS,
                "json": {
                    "symbol": "AAPL",
                    "qty": "625",
                    "side": "sell",
                    "type": "market",
                    "time_in_force": "opg",
                    "client_order_id": CID,
                },
                "timeout": 30,
            },
        )
    ]
    assert get.calls == []


@pytest.mark.parametrize(
    ("action", "side"),
    [
        (OrderAction.BUY, "buy"),
        (OrderAction.BUY_TO_COVER, "buy"),
        (OrderAction.SELL, "sell"),
        (OrderAction.SELL_SHORT, "sell"),
    ],
)
def test_side_follows_the_action(action: OrderAction, side: str) -> None:
    request = OrderRequest(order=dataclasses.replace(ORDER, action=action), client_order_id=CID)
    post = FakePost(FakeResponse(200, {"id": "ord-1"}))
    _broker(post=post).submit(request)
    assert post.calls[0][1]["json"]["side"] == side


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(ValueError("not json"), id="invalid-json"),
        pytest.param(["ord-1"], id="not-an-object"),
        pytest.param({"status": "accepted"}, id="missing-id"),
        pytest.param({"id": ""}, id="empty-id"),
        pytest.param({"id": 7}, id="non-string-id"),
    ],
)
def test_accepted_without_an_order_id_raises(payload: Any) -> None:
    with pytest.raises(BrokerError, match="POST /v2/orders"):
        _broker(post=FakePost(FakeResponse(200, payload))).submit(REQUEST)


def test_403_is_rejected_with_the_broker_message() -> None:
    post = FakePost(FakeResponse(403, {"code": 40310000, "message": "insufficient buying power"}))
    assert _broker(post=post).submit(REQUEST) == _result(
        SubmitStatus.REJECTED, reason="insufficient buying power"
    )


@pytest.mark.parametrize(
    "payload",
    [ValueError("not json"), None, ["x"], {}, {"message": 5}, {"message": ""}],
    ids=["invalid-json", "null", "array", "no-message", "non-string", "empty"],
)
def test_403_without_a_message_falls_back_to_the_status(payload: Any) -> None:
    post = FakePost(FakeResponse(403, payload))
    assert _broker(post=post).submit(REQUEST) == _result(SubmitStatus.REJECTED, reason="HTTP 403")


@pytest.mark.parametrize("status", ["new", "accepted", "pending_new", "partially_filled", "filled"])
def test_422_with_a_live_or_filled_order_is_duplicate(status: str) -> None:
    get = FakeGet(FakeResponse(200, _existing(status=status)))
    post = FakePost(FakeResponse(422, NOT_UNIQUE))
    assert _broker(get, post).submit(REQUEST) == _result(
        SubmitStatus.DUPLICATE, broker_order_id="ord-9"
    )
    assert get.calls == [
        (LOOKUP_URL, {"headers": HEADERS, "params": {"client_order_id": CID}, "timeout": 30})
    ]


@pytest.mark.parametrize("status", ["canceled", "expired", "rejected"])
def test_422_with_a_dead_order_is_rejected(status: str) -> None:
    get = FakeGet(FakeResponse(200, _existing(status=status)))
    post = FakePost(FakeResponse(422, NOT_UNIQUE))
    assert _broker(get, post).submit(REQUEST) == _result(
        SubmitStatus.REJECTED, reason=f"existing order {status}"
    )


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"code": 42210000, "message": "qty must be > 0"}, "qty must be > 0"),
        (ValueError("not json"), "HTTP 422"),
        ({}, "HTTP 422"),
    ],
    ids=["message", "invalid-json", "no-message"],
)
def test_422_without_an_existing_order_is_rejected(payload: Any, reason: str) -> None:
    get = FakeGet(FakeResponse(404, {"message": "order not found"}))
    post = FakePost(FakeResponse(422, payload))
    assert _broker(get, post).submit(REQUEST) == _result(SubmitStatus.REJECTED, reason=reason)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(ValueError("not json"), id="invalid-json"),
        pytest.param([_existing()], id="not-an-object"),
        pytest.param(_existing(client_order_id="hf-other"), id="mismatched-client-order-id"),
        pytest.param({**_existing(), "id": None}, id="null-id"),
        pytest.param(_existing(id=""), id="empty-id"),
        pytest.param(_existing(status=None), id="non-string-status"),
    ],
)
def test_422_lookup_with_a_malformed_order_raises(payload: Any) -> None:
    get = FakeGet(FakeResponse(200, payload))
    post = FakePost(FakeResponse(422, NOT_UNIQUE))
    with pytest.raises(BrokerError, match="GET /v2/orders:by_client_order_id"):
        _broker(get, post).submit(REQUEST)


@pytest.mark.parametrize("status", [201, 400, 401, 403])
def test_422_lookup_other_status_raises(status: int) -> None:
    get = FakeGet(FakeResponse(status))
    post = FakePost(FakeResponse(422, NOT_UNIQUE))
    with pytest.raises(BrokerError, match=f"GET /v2/orders:by_client_order_id: HTTP {status}"):
        _broker(get, post).submit(REQUEST)


@pytest.mark.parametrize("status", [201, 400, 401, 404])
def test_other_post_status_raises(status: int) -> None:
    get, post = FakeGet(), FakePost(FakeResponse(status, {"message": "nope"}))
    with pytest.raises(BrokerError, match=f"POST /v2/orders: HTTP {status}"):
        _broker(get, post).submit(REQUEST)
    assert len(post.calls) == 1
    assert get.calls == []


def test_post_is_retried_on_429() -> None:
    fake_time = FakeTime()
    post = FakePost(FakeResponse(429), FakeResponse(200, {"id": "ord-1"}))
    assert _broker(post=post, clock=fake_time).submit(REQUEST) == _result(
        SubmitStatus.ACCEPTED, broker_order_id="ord-1"
    )
    assert len(post.calls) == 2
    assert fake_time.sleeps == [1.0]


def test_post_gives_up_after_three_retries() -> None:
    post = FakePost(*[FakeResponse(503)] * 4)
    with pytest.raises(BrokerError, match="POST /v2/orders: HTTP 503; gave up after 3 retries"):
        _broker(post=post).submit(REQUEST)
    assert len(post.calls) == 4


def test_post_is_retried_after_a_transport_timeout() -> None:
    fake_time = FakeTime()
    post = FakePost(requests.Timeout("slow"), FakeResponse(200, {"id": "ord-1"}))
    assert _broker(post=post, clock=fake_time).submit(REQUEST) == _result(
        SubmitStatus.ACCEPTED, broker_order_id="ord-1"
    )
    assert len(post.calls) == 2
    assert fake_time.sleeps == [1.0]


@pytest.mark.parametrize("status", [429, 503])
def test_422_lookup_is_retried_on_429_and_5xx(status: int) -> None:
    get = FakeGet(FakeResponse(status), FakeResponse(200, _existing()))
    post = FakePost(FakeResponse(422, NOT_UNIQUE))
    assert _broker(get, post).submit(REQUEST) == _result(
        SubmitStatus.DUPLICATE, broker_order_id="ord-9"
    )
    assert len(get.calls) == 2
    assert len(post.calls) == 1


@pytest.mark.parametrize("status", [429, 503])
def test_422_lookup_gives_up_after_three_retries(status: int) -> None:
    get = FakeGet(*[FakeResponse(status)] * 4)
    post = FakePost(FakeResponse(422, NOT_UNIQUE))
    with pytest.raises(
        BrokerError,
        match=f"GET /v2/orders:by_client_order_id: HTTP {status}; gave up after 3 retries",
    ):
        _broker(get, post).submit(REQUEST)
    assert len(get.calls) == 4
    assert len(post.calls) == 1


def test_lookup_is_throttled_after_the_post() -> None:
    fake_time = FakeTime()
    get = FakeGet(FakeResponse(200, _existing()))
    post = FakePost(FakeResponse(422, NOT_UNIQUE))
    _broker(get, post, fake_time).submit(REQUEST)
    assert fake_time.sleeps == [pytest.approx(0.3)]


# --- order_status -----------------------------------------------------------------------

SCID = "hf-2026-09-30-AAA-buy-10"
OPEN = datetime(2026, 10, 1, 13, 30, tzinfo=UTC)
PENDING_STATUSES = [
    "new",
    "accepted",
    "pending_new",
    "accepted_for_bidding",
    "partially_filled",
    "done_for_day",
    "pending_cancel",
    "pending_replace",
    "calculated",
    "stopped",
    "suspended",
]


def _order(**overrides: Any) -> dict[str, Any]:
    return {
        "id": "ord-1",
        "client_order_id": SCID,
        "status": "filled",
        "symbol": "AAA",
        "side": "buy",
        "asset_class": "us_equity",
        "qty": "10",
        "filled_qty": "10",
        "filled_avg_price": "101.25",
        "filled_at": "2026-10-01T13:30:00Z",
        "updated_at": "2026-10-01T13:30:01Z",
        **overrides,
    }


def _order_without(key: str) -> dict[str, Any]:
    body = _order()
    del body[key]
    return body


def _status(payload: Any, status_code: int = 200) -> OrderStatus:
    return _broker(FakeGet(FakeResponse(status_code, payload))).order_status(SCID)


def test_order_status_looks_the_id_up_and_parses_a_filled_order() -> None:
    get, post = FakeGet(FakeResponse(200, _order())), FakePost()
    assert _broker(get, post).order_status(SCID) == OrderStatus(
        client_order_id=SCID,
        state=OrderState.FILLED,
        broker_order_id="ord-1",
        ticker="AAA",
        side="buy",
        filled_shares=Decimal("10"),
        avg_price=101.25,
        filled_at=OPEN,
        broker_state="filled",
    )
    assert get.calls == [
        (LOOKUP_URL, {"headers": HEADERS, "params": {"client_order_id": SCID}, "timeout": 30})
    ]
    assert post.calls == []


def test_404_is_not_found() -> None:
    assert _status({"message": "order not found"}, 404) == OrderStatus(
        client_order_id=SCID,
        state=OrderState.NOT_FOUND,
        broker_order_id=None,
        ticker=None,
        side=None,
        filled_shares=Decimal("0"),
        avg_price=None,
        filled_at=None,
        broker_state=None,
    )


@pytest.mark.parametrize("raw", ["canceled", "expired", "rejected"])
def test_a_dead_order_with_shares_is_partial_terminal(raw: str) -> None:
    status = _status(_order(status=raw, filled_qty="4"))
    assert (status.state, status.filled_shares, status.broker_state) == (
        OrderState.PARTIAL_TERMINAL,
        Decimal("4"),
        raw,
    )
    assert status.filled_at == OPEN


@pytest.mark.parametrize(
    ("raw", "state"),
    [
        ("canceled", OrderState.CANCELED),
        ("expired", OrderState.EXPIRED),
        ("rejected", OrderState.REJECTED),
    ],
)
def test_a_dead_order_without_shares_keeps_its_state(raw: str, state: OrderState) -> None:
    status = _status(_order(status=raw, filled_qty="0", filled_avg_price=None, filled_at=None))
    assert (status.state, status.filled_shares, status.avg_price, status.filled_at) == (
        state,
        Decimal("0"),
        None,
        None,
    )


@pytest.mark.parametrize("raw", PENDING_STATUSES)
def test_open_statuses_are_pending(raw: str) -> None:
    status = _status(_order(status=raw, filled_qty="0", filled_avg_price=None, filled_at=None))
    assert (status.state, status.broker_state, status.filled_at) == (OrderState.PENDING, raw, None)


@pytest.mark.parametrize("raw", ["replaced", "held", "something_new"])
def test_other_statuses_are_unrecognised_with_the_raw_status(raw: str) -> None:
    status = _status(_order(status=raw))
    assert (status.state, status.broker_state) == (OrderState.UNRECOGNISED, raw)


def test_quantities_are_decimals_and_a_null_price_is_none() -> None:
    status = _status(_order(status="new", filled_qty="2.5", filled_avg_price=None))
    assert status.filled_shares == Decimal("2.5")
    assert status.avg_price is None


@pytest.mark.parametrize(
    ("filled_at", "expected"),
    [
        pytest.param("2026-10-01T13:30:00Z", OPEN, id="z-suffix"),
        pytest.param("2026-10-01T09:30:00-04:00", OPEN, id="offset"),
        pytest.param(
            "2026-10-01T13:30:00.123456789Z",
            datetime(2026, 10, 1, 13, 30, 0, 123456, tzinfo=UTC),
            id="nine-fractional-digits",
        ),
    ],
)
def test_filled_at_is_parsed_to_utc(filled_at: str, expected: datetime) -> None:
    status = _status(_order(filled_at=filled_at))
    assert status.filled_at == expected
    assert status.filled_at is not None
    assert status.filled_at.tzinfo is UTC


def test_a_partial_terminal_order_falls_back_to_updated_at() -> None:
    status = _status(
        _order(status="expired", filled_qty="4", filled_at=None, updated_at="2026-10-01T20:00:05Z")
    )
    assert status.filled_at == datetime(2026, 10, 1, 20, 0, 5, tzinfo=UTC)


def test_a_partial_terminal_order_without_any_timestamp_raises() -> None:
    body = _order(status="canceled", filled_qty="4", filled_at=None, updated_at=None)
    with pytest.raises(
        OrderDataError, match="partial-terminal order has no filled_at or updated_at"
    ):
        _status(body)


def test_a_filled_order_without_filled_at_raises() -> None:
    with pytest.raises(OrderDataError, match="filled order has no filled_at"):
        _status(_order(filled_at=None))


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(ValueError("not json"), id="invalid-json"),
        pytest.param([_order()], id="not-an-object"),
        pytest.param(_order(client_order_id="hf-other"), id="mismatched-client-order-id"),
        pytest.param(_order(asset_class="crypto"), id="non-equity"),
        pytest.param(_order_without("asset_class"), id="missing-asset-class"),
        pytest.param(_order_without("side"), id="missing-side"),
        pytest.param(_order(side=""), id="empty-side"),
        pytest.param(_order_without("id"), id="missing-id"),
        pytest.param(_order_without("status"), id="missing-status"),
        pytest.param(_order_without("symbol"), id="missing-symbol"),
        pytest.param(_order_without("filled_qty"), id="missing-filled-qty"),
        pytest.param(_order(filled_qty=10), id="numeric-filled-qty"),
        pytest.param(_order(filled_qty="abc"), id="unparsable-filled-qty"),
        pytest.param(_order(filled_qty="nan"), id="nan-filled-qty"),
        pytest.param(_order(filled_qty="-1"), id="negative-filled-qty"),
        pytest.param(_order(filled_avg_price="abc"), id="unparsable-price"),
        pytest.param(_order(filled_avg_price=101.25), id="numeric-price"),
        pytest.param(_order(filled_at="yesterday"), id="unparsable-filled-at"),
        pytest.param(_order(filled_at="2026-10-01T13:30:00"), id="naive-filled-at"),
        pytest.param(_order(filled_at=5), id="numeric-filled-at"),
    ],
)
def test_an_unusable_body_raises_order_data_error(payload: Any) -> None:
    with pytest.raises(OrderDataError, match="GET /v2/orders:by_client_order_id"):
        _status(payload)


@pytest.mark.parametrize("status_code", [400, 401, 403, 422])
def test_other_http_codes_raise_plain_broker_error(status_code: int) -> None:
    with pytest.raises(BrokerError, match=f"HTTP {status_code}") as info:
        _status({"message": "nope"}, status_code)
    assert type(info.value) is BrokerError


def test_a_transport_failure_raises_plain_broker_error() -> None:
    get = FakeGet(*[requests.Timeout("slow")] * 4)
    with pytest.raises(BrokerError, match="gave up after 3 retries") as info:
        _broker(get).order_status(SCID)
    assert type(info.value) is BrokerError


def test_package_exports() -> None:
    import hedge_fund.adapters.broker as broker

    assert broker.__all__ == ["AlpacaPaperBroker"]
    assert broker.AlpacaPaperBroker is AlpacaPaperBroker
