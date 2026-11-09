from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pandas as pd
import pytest

from hedge_fund.adapters.broker.simulated import SimulatedBroker, SimulationError
from hedge_fund.domain.backtest import SimEvent, SimEventKind
from hedge_fund.domain.execution import OrderRequest, SubmitStatus
from hedge_fund.domain.ledger import OrderState
from hedge_fund.domain.orders import Holding, Order, OrderAction
from hedge_fund.engine.ledger.open_order import is_open
from hedge_fund.ports.broker import Broker
from tests.factories import make_frozen_data
from tests.fakes import not_found

DAYS = pd.bdate_range("2024-01-01", periods=5)
D1, D2, D3, D4, D5 = (d.date() for d in DAYS)
_NOW = datetime(2024, 1, 2, tzinfo=UTC)


def _data():
    closes = pd.DataFrame(
        {"AAA": [10.0, 11.0, 12.0, 13.0, 14.0], "BBB": [50.0, 49.0, 48.0, 47.0, 46.0]},
        index=DAYS,
    )
    return make_frozen_data(
        start=D1,
        end=D5,
        opens=closes - 0.5,
        closes=closes,
        splits=(),
        dividends=(),
    )


def _broker(cash: float = 100_000.0) -> SimulatedBroker:
    return SimulatedBroker(_data(), cash=cash)


def _request(
    cid: str = "c1",
    ticker: str = "AAA",
    shares: int = 10,
    action: OrderAction = OrderAction.BUY,
) -> OrderRequest:
    order = Order(ticker, action, shares, 10.0, 10.0 * shares)
    return OrderRequest(order=order, client_order_id=cid)


def test_simulated_broker_satisfies_the_port() -> None:
    broker: Broker = SimulatedBroker(_data(), cash=100_000.0)
    assert broker.account().equity == 100_000.0


def test_fresh_broker_has_cash_equity_and_no_positions() -> None:
    broker = _broker(12_345.0)
    assert broker.account().equity == 12_345.0
    assert broker.positions() == ()
    assert broker.events == ()


def test_equity_marks_holdings_and_positions_are_sorted_without_zeros() -> None:
    broker = _broker(1_000.0)
    broker.shares = {"BBB": -2, "AAA": 5, "ZERO": 0}
    broker.marks = {"BBB": 50.0, "AAA": 10.0, "ZERO": 1.0}
    assert broker.account().equity == 1_000.0 + 50.0 - 100.0
    assert broker.positions() == (Holding("AAA", 5), Holding("BBB", -2))


def test_submit_accepts_then_reports_duplicate_without_a_new_order() -> None:
    broker = _broker()
    first = broker.submit(_request())
    again = broker.submit(_request())
    other = broker.submit(_request("c2"))
    assert first.status is SubmitStatus.ACCEPTED
    assert first.broker_order_id == "sim-1"
    assert first.reason is None
    assert again.status is SubmitStatus.DUPLICATE
    assert again.broker_order_id == "sim-1"
    assert again.reason is None
    assert other.broker_order_id == "sim-2"
    assert len(broker.orders) == 2


@pytest.mark.parametrize(
    ("ticker", "shares", "reason"),
    [("AAA", 0, "shares"), ("AAA", -3, "shares"), ("ZZZ", 5, "ZZZ")],
)
def test_submit_rejects_bad_requests(ticker: str, shares: int, reason: str) -> None:
    broker = _broker()
    result = broker.submit(_request("bad", ticker, shares))
    assert result.status is SubmitStatus.REJECTED
    assert result.broker_order_id is None
    assert result.reason is not None
    assert reason in result.reason
    assert broker.orders == {}
    assert broker.order_status("bad") == not_found("bad")
    (event,) = broker.events
    assert event == SimEvent(
        day=event.day,
        kind=SimEventKind.REJECT,
        ticker=ticker,
        shares=0,
        price=0.0,
        cash=0.0,
        client_order_id="bad",
        reason=result.reason,
    )


def test_reject_event_carries_the_current_day() -> None:
    broker = _broker()
    broker.open(D1)
    broker.submit(_request("bad", shares=0))
    assert broker.events[0].day == D1


def test_pending_status_shape() -> None:
    broker = _broker()
    broker.submit(_request("c1", "BBB", 7, OrderAction.SELL_SHORT))
    status = broker.order_status("c1")
    assert status.client_order_id == "c1"
    assert status.state is OrderState.PENDING
    assert status.broker_order_id == "sim-1"
    assert status.ticker == "BBB"
    assert status.side == "sell"
    assert status.filled_shares == Decimal(0)
    assert status.avg_price is None
    assert status.filled_at is None
    assert status.broker_state == "pending"


@pytest.mark.parametrize(
    ("action", "side"),
    [
        (OrderAction.BUY, "buy"),
        (OrderAction.BUY_TO_COVER, "buy"),
        (OrderAction.SELL, "sell"),
        (OrderAction.SELL_SHORT, "sell"),
    ],
)
def test_status_side_follows_the_action(action: OrderAction, side: str) -> None:
    broker = _broker()
    broker.submit(_request(action=action))
    assert broker.order_status("c1").side == side


def test_status_of_a_filled_order_reports_the_fill_at_the_open_in_utc() -> None:
    broker = _broker()
    broker.submit(_request("c1", "AAA", 10))
    sim = broker.orders["c1"]
    sim.state, sim.fill_price, sim.filled_on = OrderState.FILLED, 10.05, D1
    status = broker.order_status("c1")
    assert status.state is OrderState.FILLED
    assert status.filled_shares == Decimal(10)
    assert status.avg_price == 10.05
    assert status.filled_at == datetime(2024, 1, 1, 14, 30, tzinfo=UTC)
    assert status.broker_state == "filled"
    assert not is_open(
        status.state, acknowledged=True, deadline=datetime(2030, 1, 1, tzinfo=UTC), now=_NOW
    )


def test_status_of_an_order_rejected_at_fill_time_has_no_fill() -> None:
    broker = _broker()
    broker.submit(_request("c1"))
    broker.orders["c1"].state = OrderState.REJECTED
    status = broker.order_status("c1")
    assert status.state is OrderState.REJECTED
    assert (status.filled_shares, status.avg_price, status.filled_at) == (Decimal(0), None, None)
    assert status.broker_state == "rejected"


def test_unknown_order_is_not_found() -> None:
    assert _broker().order_status("nope") == not_found("nope")


def test_pending_order_is_open() -> None:
    broker = _broker()
    broker.submit(_request())
    now = datetime(2024, 1, 2, tzinfo=UTC)
    state = broker.order_status("c1").state
    assert is_open(state, acknowledged=True, deadline=now + timedelta(hours=1), now=now)


def test_orders_remember_the_submission_day() -> None:
    broker = _broker()
    broker.submit(_request("early"))
    broker.open(D1)
    broker.submit(_request("late"))
    assert [o.submitted_on for o in broker.orders.values()] == [None, D1]
    assert list(broker.orders) == ["early", "late"]


def test_day_bookkeeping() -> None:
    broker = _broker()
    assert broker.day is None
    broker.open(D1)
    assert (broker.day, broker.phase) == (D1, "open")
    broker.close(D1)
    assert (broker.day, broker.phase) == (D1, "closed")
    broker.open(D3)
    assert (broker.day, broker.phase) == (D3, "open")


def test_open_twice_is_an_error() -> None:
    broker = _broker()
    broker.open(D1)
    with pytest.raises(SimulationError, match="after the last day"):
        broker.open(D1)


def test_close_before_open_is_an_error() -> None:
    with pytest.raises(SimulationError, match="not open"):
        _broker().close(D1)


def test_close_of_another_day_is_an_error() -> None:
    broker = _broker()
    broker.open(D1)
    with pytest.raises(SimulationError, match="not open"):
        broker.close(D2)


def test_close_twice_is_an_error() -> None:
    broker = _broker()
    broker.open(D1)
    broker.close(D1)
    with pytest.raises(SimulationError, match="not open"):
        broker.close(D1)


def test_open_of_a_date_outside_the_index_is_an_error() -> None:
    with pytest.raises(SimulationError, match="not a trading day"):
        _broker().open(date(2024, 1, 6))


def test_open_out_of_order_is_an_error() -> None:
    broker = _broker()
    broker.open(D3)
    broker.close(D3)
    with pytest.raises(SimulationError, match="after the last day"):
        broker.open(D2)


def test_open_without_closing_the_previous_day_is_an_error() -> None:
    broker = _broker()
    broker.open(D1)
    with pytest.raises(SimulationError, match="not closed"):
        broker.open(D2)


@pytest.mark.parametrize("cash", [0.0, -1.0, float("nan"), float("inf")])
def test_constructor_rejects_bad_cash(cash: float) -> None:
    with pytest.raises(SimulationError, match="cash must be finite and > 0"):
        SimulatedBroker(_data(), cash=cash)
