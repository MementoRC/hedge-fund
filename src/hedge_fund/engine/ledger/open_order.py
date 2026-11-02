"""Whether one broker answer leaves an order live: the pending cases of settle's lookup.

Pure: the instant and the deadline are passed in.
"""

from __future__ import annotations

from datetime import datetime

from hedge_fund.domain.ledger import OrderState

_OPEN = frozenset({OrderState.PENDING, OrderState.UNRECOGNISED})


def is_open(state: OrderState, *, acknowledged: bool, deadline: datetime, now: datetime) -> bool:
    """True if the order may still fill: PENDING, UNRECOGNISED (fail closed), or NOT_FOUND
    unacknowledged before ``deadline`` (the broker may not have listed it yet).

    An acknowledged NOT_FOUND is not open: settle reports it unsettled, never pending.
    Every final state (filled, partial-terminal, canceled, rejected, expired) is not open.
    """
    if state is OrderState.NOT_FOUND:
        return not acknowledged and now < deadline
    return state in _OPEN
