"""nav_history: one NAV point per date from the latest read, future reads dropped."""

from __future__ import annotations

import dataclasses
from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from hedge_fund.domain.account import AccountSnapshot
from hedge_fund.domain.orders import Holding
from hedge_fund.domain.risk import NavPoint
from hedge_fund.engine.account.history import nav_history

AS_OF = date(2026, 10, 23)
EASTERN = timezone(timedelta(hours=-4))


def _snap(
    day: date, hour: int, equity: float, *, holdings: tuple[Holding, ...] = ()
) -> AccountSnapshot:
    """A read on ``day`` at ``hour``:00 UTC."""
    return AccountSnapshot(
        as_of=day,
        observed_at=datetime(day.year, day.month, day.day, hour, tzinfo=UTC),
        equity=equity,
        holdings=holdings,
    )


def _day(offset: int) -> date:
    return AS_OF + timedelta(days=offset)


def test_one_point_per_date_ascending() -> None:
    snapshots = [_snap(_day(0), 13, 103.0), _snap(_day(-2), 13, 101.0), _snap(_day(-1), 13, 102.0)]
    assert nav_history(snapshots, as_of=AS_OF) == (
        NavPoint(date=_day(-2), nav=101.0),
        NavPoint(date=_day(-1), nav=102.0),
        NavPoint(date=_day(0), nav=103.0),
    )


@pytest.mark.parametrize("reverse", [False, True], ids=["oldest-first", "newest-first"])
def test_latest_observed_at_wins_within_a_date(reverse: bool) -> None:
    reads = [_snap(AS_OF, 12, 100.0), _snap(AS_OF, 14, 98.0), _snap(AS_OF, 13, 99.0)]
    snapshots = list(reversed(reads)) if reverse else reads
    assert nav_history(snapshots, as_of=AS_OF) == (NavPoint(date=AS_OF, nav=98.0),)


def test_latest_is_by_instant_not_by_wall_clock() -> None:
    # 10:00 at UTC-4 is 14:00 UTC: later than the 13:00 UTC read despite the smaller hour.
    eastern = dataclasses.replace(
        _snap(AS_OF, 0, 97.0), observed_at=datetime(2026, 10, 23, 10, tzinfo=EASTERN)
    )
    snapshots = [eastern, _snap(AS_OF, 13, 99.0)]
    assert nav_history(snapshots, as_of=AS_OF) == (NavPoint(date=AS_OF, nav=97.0),)


def test_reads_dated_after_as_of_are_dropped() -> None:
    snapshots = [_snap(_day(-1), 13, 100.0), _snap(_day(0), 13, 99.0), _snap(_day(1), 13, 50.0)]
    assert nav_history(snapshots, as_of=AS_OF) == (
        NavPoint(date=_day(-1), nav=100.0),
        NavPoint(date=_day(0), nav=99.0),
    )


def test_nothing_on_or_before_as_of_gives_an_empty_history() -> None:
    assert nav_history([_snap(_day(1), 13, 100.0)], as_of=AS_OF) == ()


def test_empty_input_gives_an_empty_history() -> None:
    assert nav_history([], as_of=AS_OF) == ()


def test_equal_observed_at_with_different_content_raises() -> None:
    first = _snap(AS_OF, 13, 100.0)
    second = _snap(AS_OF, 13, 100.0, holdings=(Holding(ticker="AAPL", shares=1),))
    with pytest.raises(ValueError, match="two different snapshots for 2026-10-23 observed at"):
        nav_history([first, second], as_of=AS_OF)


def test_conflict_raises_even_when_a_later_read_would_win() -> None:
    later = _snap(AS_OF, 15, 97.0)
    conflict = [_snap(AS_OF, 13, 100.0), _snap(AS_OF, 13, 101.0)]
    for snapshots in ([later, *conflict], [*conflict, later]):
        with pytest.raises(ValueError, match="two different snapshots"):
            nav_history(snapshots, as_of=AS_OF)


def test_identical_duplicates_collapse() -> None:
    read = _snap(AS_OF, 13, 100.0, holdings=(Holding(ticker="AAPL", shares=1),))
    copy = dataclasses.replace(read)
    assert nav_history([read, copy, read], as_of=AS_OF) == (NavPoint(date=AS_OF, nav=100.0),)


@pytest.mark.parametrize("day", [_day(0), _day(1)], ids=["on-as_of", "after-as_of"])
def test_an_invalid_snapshot_fails_closed(day: date) -> None:
    snapshots = [_snap(_day(-1), 13, 100.0), _snap(day, 13, 0.0)]
    with pytest.raises(ValueError, match="nav must be a finite number > 0"):
        nav_history(snapshots, as_of=AS_OF)


def test_package_exports() -> None:
    import hedge_fund.engine.account as account
    from hedge_fund.engine.account.validate import validate_snapshot

    assert account.__all__ == ["nav_history", "validate_snapshot"]
    assert account.nav_history is nav_history
    assert account.validate_snapshot is validate_snapshot
