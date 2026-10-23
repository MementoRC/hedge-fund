"""Account snapshot checks: a read is never saved or folded into NAV unless it passes.

Pure: no I/O. The equity rules are ``validate_nav``'s, applied to a one-point history.
"""

from __future__ import annotations

import itertools

from hedge_fund.domain.account import AccountSnapshot
from hedge_fund.domain.risk import NavPoint
from hedge_fund.engine.risk.breakers import validate_nav


def validate_snapshot(snapshot: AccountSnapshot) -> None:
    """Raise ValueError unless ``snapshot`` is a usable read.

    ``equity`` must pass ``validate_nav`` (a non-bool int or float, finite, > 0);
    ``observed_at`` must be tz-aware (any offset); ``holdings`` must be strictly ascending
    by ticker, each with non-zero, non-bool int ``shares``. An empty book is valid.
    """
    validate_nav((NavPoint(date=snapshot.as_of, nav=snapshot.equity),))
    observed_at = snapshot.observed_at
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError(f"observed_at must be tz-aware, got {observed_at!r}")
    for holding in snapshot.holdings:
        shares = holding.shares
        if isinstance(shares, bool) or not isinstance(shares, int) or shares == 0:
            raise ValueError(f"{holding.ticker!r} shares must be a non-zero int, got {shares!r}")
    for before, after in itertools.pairwise(snapshot.holdings):
        if not before.ticker < after.ticker:
            raise ValueError(
                "holdings must be strictly ascending by ticker, "
                f"got {after.ticker!r} after {before.ticker!r}"
            )
