"""is_open: which broker answers leave an order live (settle's pending and fail-closed cases)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from hedge_fund.domain.ledger import OrderState
from hedge_fund.engine.ledger.open_order import is_open

DEADLINE = datetime(2026, 10, 1, 20, 0, tzinfo=UTC)
BEFORE = DEADLINE - timedelta(minutes=1)
AFTER = DEADLINE + timedelta(minutes=1)

_DECIDED_BY_STATE = {
    OrderState.PENDING: True,
    OrderState.UNRECOGNISED: True,  # fail closed
    OrderState.FILLED: False,
    OrderState.PARTIAL_TERMINAL: False,
    OrderState.CANCELED: False,
    OrderState.REJECTED: False,
    OrderState.EXPIRED: False,
}


def test_every_state_but_not_found_is_decided_by_the_state_alone() -> None:
    assert set(_DECIDED_BY_STATE) == set(OrderState) - {OrderState.NOT_FOUND}


@pytest.mark.parametrize("acknowledged", [True, False])
@pytest.mark.parametrize("now", [BEFORE, AFTER])
@pytest.mark.parametrize(("state", "expected"), list(_DECIDED_BY_STATE.items()))
def test_a_found_order_is_open_only_when_pending_or_unrecognised(
    state: OrderState, expected: bool, now: datetime, acknowledged: bool
) -> None:
    assert is_open(state, acknowledged=acknowledged, deadline=DEADLINE, now=now) is expected


@pytest.mark.parametrize(
    ("acknowledged", "now", "expected"),
    [
        pytest.param(False, BEFORE, True, id="unacknowledged-before-the-deadline"),
        pytest.param(False, DEADLINE, False, id="unacknowledged-at-the-deadline"),
        pytest.param(False, AFTER, False, id="unacknowledged-after-the-deadline"),
        pytest.param(True, BEFORE, False, id="acknowledged-before-the-deadline"),
        pytest.param(True, AFTER, False, id="acknowledged-after-the-deadline"),
    ],
)
def test_a_not_found_order_is_open_only_if_unacknowledged_before_its_deadline(
    acknowledged: bool, now: datetime, expected: bool
) -> None:
    result = is_open(OrderState.NOT_FOUND, acknowledged=acknowledged, deadline=DEADLINE, now=now)
    assert result is expected
