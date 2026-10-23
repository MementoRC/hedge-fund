"""Account snapshot: broker equity and positions read together on one run."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from hedge_fund.domain.orders import Holding


@dataclass(frozen=True)
class AccountSnapshot:
    """One broker read for the run dated ``as_of``.

    ``observed_at`` is the tz-aware instant of the read, ``equity`` is ``Account.equity``
    (recorded as NAV) and ``holdings`` is ``Broker.positions()``: signed, sorted by ticker.
    Checked by ``engine.account.validate_snapshot``.
    """

    as_of: date
    observed_at: datetime
    equity: float
    holdings: tuple[Holding, ...]
