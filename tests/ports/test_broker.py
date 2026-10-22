"""Tests for the Broker port, exercised through the scripted fake."""

from __future__ import annotations

import pytest

from hedge_fund.domain.execution import Account, OrderRequest, SubmitResult, SubmitStatus
from hedge_fund.domain.orders import Holding, Order, OrderAction
from hedge_fund.ports.broker import Broker, BrokerError
from tests.fakes import SCRIPTED_REJECTION, ScriptedBroker

CID = "hf-2026-10-21-A-sell-5"
ORDER = Order(ticker="A", action=OrderAction.SELL, shares=5, ref_price=10.0, notional=50.0)
REQUEST = OrderRequest(order=ORDER, client_order_id=CID)


def test_broker_error_is_an_exception() -> None:
    assert issubclass(BrokerError, Exception)


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


def test_scripted_broker_raises_scripted_exceptions() -> None:
    fake = ScriptedBroker({CID: BrokerError("down")})
    with pytest.raises(BrokerError, match="down"):
        fake.submit(REQUEST)
    assert fake.submitted == [CID]
