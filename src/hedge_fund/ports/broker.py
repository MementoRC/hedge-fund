"""Broker port: account equity, signed holdings and order submission."""

from __future__ import annotations

from typing import Protocol

from hedge_fund.domain.execution import Account, OrderRequest, SubmitResult
from hedge_fund.domain.orders import Holding


class BrokerError(Exception):
    """A transport, auth, status or payload failure while talking to the broker."""


class OrderDataError(BrokerError):
    """One order's 200 body is unusable: unparsable JSON, a missing or invalid field, another
    order's ``client_order_id``, a non-equity asset, or a filled order with no timestamp.

    Settle records it against that order and carries on; a plain BrokerError (transport,
    auth, unexpected HTTP status) stops the run.
    """


class Broker(Protocol):
    """Paper-trading broker. Implementations live in adapters/broker/.

    Every method raises BrokerError on a transport or auth failure after retries, an
    unexpected status, or a malformed payload.
    """

    def account(self) -> Account:
        """Current account; ``equity`` is what 6d records as NAV."""

    def positions(self) -> tuple[Holding, ...]:
        """Signed whole shares (+ long, - short), sorted by ticker, one entry per ticker.

        Tickers with zero shares are left out.
        """

    def submit(self, request: OrderRequest) -> SubmitResult:
        """Place a market order for the opening auction.

        Returns a result whose status is ACCEPTED, DUPLICATE (an order with this
        client_order_id already exists) or REJECTED. On BrokerError the outcome is unknown;
        re-submitting the same request is safe.
        """
