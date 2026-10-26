"""Tests for the Broker port, exercised through the scripted fake."""

from __future__ import annotations

from decimal import Decimal

import pytest

from hedge_fund.domain.execution import Account, OrderRequest, SubmitResult, SubmitStatus
from hedge_fund.domain.ledger import OrderState, OrderStatus
from hedge_fund.domain.orders import Holding, Order, OrderAction
from hedge_fund.ports.broker import Broker, BrokerError, OrderDataError
from tests.fakes import SCRIPTED_REJECTION, ScriptedBroker, not_found

CID = "hf-2026-10-21-A-sell-5"
ORDER = Order(ticker="A", action=OrderAction.SELL, shares=5, ref_price=10.0, notional=50.0)
REQUEST = OrderRequest(order=ORDER, client_order_id=CID)


def test_broker_error_is_an_exception() -> None:
    assert issubclass(BrokerError, Exception)


def test_order_data_error_is_a_broker_error() -> None:
    assert issubclass(OrderDataError, BrokerError)


def test_scripted_broker_satisfies_the_port() -> None:
    fake = ScriptedBroker(equity=250.0, holdings=(Holding(ticker="A", shares=-5),))
    broker: Broker = fake
    assert broker.account() == Account(equity=250.0)
    assert broker.positions() == (Holding(ticker="A", shares=-5),)
    assert broker.submit(REQUEST) == SubmitResult(
        request=REQUEST, status=SubmitStatus.ACCEPTED, broker_order_id=f"b-{CID}", reason=None
    )
    assert fake.submitted == [CID]


@pytest.mark.parametrize(
    ("status", "broker_order_id", "reason"),
    [
        (SubmitStatus.DUPLICATE, f"b-{CID}", None),
        (SubmitStatus.REJECTED, None, SCRIPTED_REJECTION),
    ],
)
def test_scripted_statuses(
    status: SubmitStatus, broker_order_id: str | None, reason: str | None
) -> None:
    result = ScriptedBroker({CID: status}).submit(REQUEST)
    assert result == SubmitResult(
        request=REQUEST, status=status, broker_order_id=broker_order_id, reason=reason
    )


@pytest.mark.parametrize(
    "status", [SubmitStatus.ERROR, SubmitStatus.DEFERRED, SubmitStatus.SKIPPED]
)
def test_scripted_broker_rejects_statuses_a_real_broker_never_returns(
    status: SubmitStatus,
) -> None:
    with pytest.raises(ValueError, match=status.name):
        ScriptedBroker({CID: status})


def test_scripted_broker_raises_scripted_read_errors() -> None:
    fake = ScriptedBroker(
        account_error=BrokerError("account down"), positions_error=BrokerError("positions down")
    )
    with pytest.raises(BrokerError, match="account down"):
        fake.account()
    with pytest.raises(BrokerError, match="positions down"):
        fake.positions()
    fake.account_error = None
    assert fake.account() == Account(equity=1_000_000.0)


def test_scripted_broker_raises_scripted_exceptions() -> None:
    fake = ScriptedBroker({CID: BrokerError("down")})
    with pytest.raises(BrokerError, match="down"):
        fake.submit(REQUEST)
    assert fake.submitted == [CID]


FILLED = OrderStatus(
    client_order_id=CID,
    state=OrderState.FILLED,
    broker_order_id="b-1",
    ticker="A",
    side="sell",
    filled_shares=Decimal("5"),
    avg_price=10.0,
    filled_at=None,
    broker_state="filled",
)


def test_scripted_order_status_returns_the_script_or_not_found() -> None:
    fake = ScriptedBroker(orders={CID: FILLED})
    broker: Broker = fake
    assert broker.order_status(CID) is FILLED
    assert broker.order_status("hf-other") == not_found("hf-other")
    assert not_found("hf-other").state is OrderState.NOT_FOUND
    assert not_found("hf-other").filled_shares == Decimal("0")
    assert fake.looked_up == [CID, "hf-other"]
    assert fake.submitted == []


def test_scripted_order_status_raises_scripted_exceptions() -> None:
    fake = ScriptedBroker(orders={CID: BrokerError("down")})
    with pytest.raises(BrokerError, match="down"):
        fake.order_status(CID)
    assert fake.looked_up == [CID]
