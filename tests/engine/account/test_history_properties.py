"""Seeded invariants of nav_history over random snapshot sets."""

from __future__ import annotations

import random
from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from hedge_fund.domain.account import AccountSnapshot
from hedge_fund.domain.orders import Holding
from hedge_fund.engine.account import nav_history
from hedge_fund.engine.risk.breakers import validate_nav

SEEDS = range(300)
AS_OF = date(2026, 10, 23)
FIRST_DAY = AS_OF - timedelta(days=6)
TICKERS = ("AAPL", "JPM", "KO", "MSFT", "XOM")
ZONES = (UTC, timezone(timedelta(hours=-4)), timezone(timedelta(hours=9)))
EPOCH = datetime(2026, 10, 1, tzinfo=UTC)


def _snapshots(seed: int) -> list[AccountSnapshot]:
    """Up to 12 reads over 9 days (the last 2 after AS_OF), every instant distinct.

    About a third of the sets also repeat one read exactly, which must collapse.
    """
    rng = random.Random(seed)
    minutes = rng.sample(range(40_000), rng.randint(0, 12))
    snapshots = []
    for minute in minutes:
        tickers = sorted(rng.sample(TICKERS, rng.randint(0, len(TICKERS))))
        snapshots.append(
            AccountSnapshot(
                as_of=FIRST_DAY + timedelta(days=rng.randint(0, 8)),
                observed_at=(EPOCH + timedelta(minutes=minute)).astimezone(rng.choice(ZONES)),
                equity=rng.uniform(50.0, 150.0),
                holdings=tuple(
                    Holding(ticker=t, shares=rng.choice([-1, 1]) * rng.randint(1, 500))
                    for t in tickers
                ),
            )
        )
    if snapshots and rng.random() < 0.35:
        snapshots.append(rng.choice(snapshots))
    rng.shuffle(snapshots)
    return snapshots


def _expected(snapshots: list[AccountSnapshot]) -> dict[date, float]:
    """Each date on or before AS_OF -> the equity of its latest read (brute force)."""
    days = {s.as_of for s in snapshots if s.as_of <= AS_OF}
    return {
        day: max((s for s in snapshots if s.as_of == day), key=lambda s: s.observed_at).equity
        for day in days
    }


@pytest.mark.parametrize("seed", SEEDS)
def test_invariant_to_input_order(seed: int) -> None:
    snapshots = _snapshots(seed)
    shuffled = random.Random(seed + 1_000).sample(snapshots, len(snapshots))
    assert nav_history(shuffled, as_of=AS_OF) == nav_history(snapshots, as_of=AS_OF)
    assert nav_history(reversed(snapshots), as_of=AS_OF) == nav_history(snapshots, as_of=AS_OF)


@pytest.mark.parametrize("seed", SEEDS)
def test_output_is_a_valid_nav_history(seed: int) -> None:
    history = nav_history(_snapshots(seed), as_of=AS_OF)
    validate_nav(history)
    assert all(point.date <= AS_OF for point in history)


@pytest.mark.parametrize("seed", SEEDS)
def test_each_date_takes_its_latest_read(seed: int) -> None:
    snapshots = _snapshots(seed)
    history = nav_history(snapshots, as_of=AS_OF)
    assert {point.date: point.nav for point in history} == _expected(snapshots)


def test_grid_exercises_every_case() -> None:
    sets = [_snapshots(seed) for seed in SEEDS]
    assert any(not snapshots for snapshots in sets)
    assert any(len(set(snapshots)) < len(snapshots) for snapshots in sets)  # duplicates
    assert any(any(s.as_of > AS_OF for s in snapshots) for snapshots in sets)  # future reads
    assert any(  # a date read more than once, so "latest" is decided
        len({s.observed_at for s in snapshots if s.as_of == day}) > 1
        for snapshots in sets
        for day in {s.as_of for s in snapshots}
    )
    assert any(nav_history(snapshots, as_of=AS_OF) == () for snapshots in sets if snapshots)
