"""Layer 6d account snapshots: record one broker read per run and fold them into NAV.

Feeding the snapshot to ``generate_orders`` and the NAV history to ``assess_portfolio`` is
the 6e pipeline's job.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime

from hedge_fund.domain.account import AccountSnapshot
from hedge_fund.engine.account import validate_snapshot
from hedge_fund.ports.account import SnapshotStore
from hedge_fund.ports.broker import Broker


def record_snapshot(
    broker: Broker,
    store: SnapshotStore,
    *,
    as_of: date,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> AccountSnapshot:
    """Read the account, then the positions; validate, save and return the snapshot.

    ``observed_at`` is ``now()``, taken after both reads. A BrokerError from either read or
    a ValueError from ``validate_snapshot`` propagates, and nothing is saved. ``as_of`` is
    not compared with ``observed_at.date()``: a pre-open run and a UTC offset make them
    legitimately differ.
    """
    equity = broker.account().equity
    holdings = broker.positions()
    snapshot = AccountSnapshot(as_of=as_of, observed_at=now(), equity=equity, holdings=holdings)
    validate_snapshot(snapshot)
    store.save(snapshot)
    return snapshot
