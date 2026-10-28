"""fill_deadline: 16:00 New York on the first weekday after the start's New York date, in UTC."""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from hedge_fund.engine.ledger.deadline import fill_deadline

NEW_YORK = ZoneInfo("America/New_York")


def _utc(month: int, day: int, hour: int) -> datetime:
    return datetime(2026, month, day, hour, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("started_at", "deadline"),
    [
        pytest.param(_utc(9, 28, 22), _utc(9, 29, 20), id="monday-evening-to-tuesday"),
        pytest.param(_utc(9, 28, 12), _utc(9, 29, 20), id="monday-pre-open-to-tuesday"),
        pytest.param(_utc(9, 30, 22), _utc(10, 1, 20), id="wednesday-to-thursday"),
        pytest.param(_utc(10, 2, 22), _utc(10, 5, 20), id="friday-to-monday"),
        pytest.param(_utc(10, 3, 14), _utc(10, 5, 20), id="saturday-to-monday"),
        pytest.param(_utc(10, 4, 14), _utc(10, 5, 20), id="sunday-to-monday"),
        pytest.param(_utc(9, 29, 2), _utc(9, 29, 20), id="utc-tuesday-is-new-york-monday"),
        pytest.param(_utc(10, 30, 22), _utc(11, 2, 21), id="edt-friday-to-est-monday"),
        pytest.param(_utc(3, 6, 23), _utc(3, 9, 20), id="est-friday-to-edt-monday"),
        pytest.param(_utc(12, 2, 23), _utc(12, 3, 21), id="winter-wednesday-to-thursday"),
        pytest.param(
            datetime(2026, 9, 30, 18, 0, tzinfo=NEW_YORK), _utc(10, 1, 20), id="new-york-input"
        ),
    ],
)
def test_deadline(started_at: datetime, deadline: datetime) -> None:
    assert fill_deadline(started_at) == deadline


def test_the_deadline_is_in_utc() -> None:
    assert fill_deadline(_utc(9, 30, 22)).tzinfo is UTC


def test_a_naive_start_raises() -> None:
    with pytest.raises(ValueError, match="must be timezone-aware"):
        fill_deadline(datetime(2026, 9, 30, 22, 0))
