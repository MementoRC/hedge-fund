"""The demo world: one seed for the prices, facts that pass the gates, the same store as before."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pandas.testing as pdt

from hedge_fund.adapters.data.frozen_store import MANIFEST, write_store
from hedge_fund.demo import facts, world
from tests import factories

GOLDEN_STORE_DIGEST = "6fa8739dad872645e7752901a53d601558d8919287e197171ee7dfba864d0bcd"


def test_the_default_seed_is_the_one_the_golden_test_uses() -> None:
    assert world.DEFAULT_SEED == 1
    default, explicit = world.make_golden_data(), world.make_golden_data(1)
    pdt.assert_frame_equal(default.closes, explicit.closes)
    pdt.assert_frame_equal(default.opens, explicit.opens)


def test_the_same_seed_gives_the_same_store() -> None:
    first, second = world.make_golden_data(7), world.make_golden_data(7)
    pdt.assert_frame_equal(first.closes, second.closes)
    pdt.assert_frame_equal(first.facts, second.facts)


def test_another_seed_moves_the_prices_and_the_facts() -> None:
    one, two = world.make_golden_data(1), world.make_golden_data(2)
    assert not one.closes.equals(two.closes)
    assert not one.facts.equals(two.facts)


def test_seed_one_is_byte_identical_to_the_golden_store(tmp_path: Path) -> None:
    # the pin from tests/services/test_backtest_golden.py (GOLDEN_STORE_DIGEST, frozen at
    # 2024-06-01 UTC): the move and the seed - 1 facts stream changed no byte of the seed-1 store
    write_store(tmp_path / "store", world.make_golden_data(), created_at=world.FROZEN_AT)
    manifest = (tmp_path / "store" / MANIFEST).read_bytes()
    assert hashlib.sha256(manifest).hexdigest() == GOLDEN_STORE_DIGEST


def test_every_seed_keeps_the_events() -> None:
    data = world.make_golden_data(2)
    stopped = data.closes.index > pd.Timestamp(world.LAST_TRADED)
    assert data.closes.loc[stopped, world.DELIST_TICKER].isna().all()
    assert data.closes.loc[~stopped, world.DELIST_TICKER].notna().all()
    assert {span.ticker for span in data.membership} == set(world.SECTORS)


def test_the_run_window_sits_inside_the_store() -> None:
    assert world.FIRST_DAY < world.RUN_START < world.LAST_DAY
    assert world.FROZEN_AT.tzinfo is not None


def test_passing_facts_have_two_filings_of_every_gate_metric_per_name() -> None:
    names = ["AAA", "BBB"]
    rows = facts.make_passing_facts(names)
    assert len(rows) == len(names) * 2 * len(facts.PASSING_CURRENT)


def test_tests_factories_reexports_the_moved_fixtures() -> None:
    assert factories.make_passing_facts is facts.make_passing_facts
    assert factories.make_passing_fundamentals is facts.make_passing_fundamentals
    assert factories.make_fundamentals is facts.make_fundamentals
    assert factories.FUNDAMENTALS_AS_OF is facts.FUNDAMENTALS_AS_OF
