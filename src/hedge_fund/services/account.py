"""Layer 6d account snapshots: record one broker read per run and fold them into NAV.

``services/cycle.py`` feeds the snapshot to ``generate_orders`` and the NAV history to the
circuit breakers and ``assess_portfolio``.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime

from hedge_fund.domain.account import AccountSnapshot
from hedge_fund.domain.risk import NavHistory
from hedge_fund.engine.account import nav_history, validate_snapshot
from hedge_fund.ports.account import SnapshotStore
from hedge_fund.ports.broker import Broker


def read_snapshot(
    broker: Broker,
    *,
    as_of: date,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> AccountSnapshot:
    """Read the account, then the positions; validate and return the snapshot (not saved).

    ``observed_at`` is ``now()``, taken after both reads. A BrokerError from either read or
    a ValueError from ``validate_snapshot`` propagates. ``as_of`` is not compared with
    ``observed_at.date()``: a pre-open run and a UTC offset make them legitimately differ.
    """
    equity = broker.account().equity
    holdings = broker.positions()
    snapshot = AccountSnapshot(as_of=as_of, observed_at=now(), equity=equity, holdings=holdings)
    validate_snapshot(snapshot)
    return snapshot


def record_snapshot(
    broker: Broker,
    store: SnapshotStore,
    *,
    as_of: date,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> AccountSnapshot:
    """``read_snapshot``, then save and return the snapshot; nothing is saved on an error."""
    snapshot = read_snapshot(broker, as_of=as_of, now=now)
    store.save(snapshot)
    return snapshot


def load_nav_history(store: SnapshotStore, *, as_of: date) -> NavHistory:
    """Every stored snapshot folded by ``nav_history``, for ``assess_portfolio(nav=...)``.

    SnapshotStoreError from the store and ValueError from ``nav_history`` propagate.
    """
    return nav_history(store.snapshots(), as_of=as_of)
