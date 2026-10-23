"""Fold account snapshots into a NAV history: one point per date, the latest read wins.

Pure: no I/O, no clock. ``services/account.py`` reads the snapshots from a SnapshotStore.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime

from hedge_fund.domain.account import AccountSnapshot
from hedge_fund.domain.risk import NavHistory, NavPoint
from hedge_fund.engine.account.validate import validate_snapshot


def nav_history(snapshots: Iterable[AccountSnapshot], *, as_of: date) -> NavHistory:
    """One ``NavPoint(date, equity)`` per snapshot date on or before ``as_of``, ascending.

    Every snapshot passes ``validate_snapshot`` first, later-dated ones included, so garbage
    from a store fails closed. Within a date the greatest ``observed_at`` wins. Two
    snapshots of one date with equal ``observed_at`` and different content raise
    ValueError; identical ones collapse. The result does not depend on input order and
    passes ``validate_nav``; it is ``()`` when nothing is dated on or before ``as_of``.
    """
    by_date: dict[date, dict[datetime, AccountSnapshot]] = {}
    for snapshot in snapshots:
        validate_snapshot(snapshot)
        if snapshot.as_of > as_of:
            continue
        reads = by_date.setdefault(snapshot.as_of, {})
        seen = reads.setdefault(snapshot.observed_at, snapshot)
        if seen != snapshot:
            raise ValueError(
                f"two different snapshots for {snapshot.as_of} observed at "
                f"{snapshot.observed_at.isoformat()}"
            )
    return tuple(
        NavPoint(date=day, nav=reads[max(reads)].equity) for day, reads in sorted(by_date.items())
    )
