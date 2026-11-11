"""Simulated-broker port: a Broker the replay drives day by day and whose events it reads."""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

from hedge_fund.domain.backtest import SimEvent
from hedge_fund.ports.broker import Broker


@runtime_checkable
class SimBroker(Broker, Protocol):
    """A Broker that simulates the market. Implementations live in adapters/broker/."""

    @property
    def events(self) -> tuple[SimEvent, ...]:
        """Everything the broker did so far, in order."""

    def open(self, day: date) -> None:
        """Start ``day``: ex-date events, then fills of orders queued on earlier days."""

    def close(self, day: date) -> None:
        """End ``day``: mark holdings to the close and accrue borrow."""
