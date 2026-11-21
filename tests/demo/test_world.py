"""The demo world: one seed for the prices, facts that pass the gates, the same store as before."""

from __future__ import annotations

import hashlib
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt

from hedge_fund.adapters.data.frozen_store import MANIFEST, write_store
from hedge_fund.demo import facts, world
from hedge_fund.domain.backtest import Dividend, FrozenData
from hedge_fund.domain.ledger import Split
from hedge_fund.domain.macro import MACRO_SERIES
from tests import factories

SAMPLE_SPY_FIRST_CLOSE = 404.264766
SAMPLE_AE1_FIRST_OPEN = 102.187122
GOLDEN_STORE_DIGEST = "71335b8226872f18b2091540649fa93b0cac05b83e5ea7f252d23c27f903a7ea"
# sha256 of seed 1's opens, closes and dividends, taken before the demo gained macro series:
# the macro stream is seeded apart, so these bytes must never move
SEED_ONE_PRICE_DIGESTS = {
    "opens": "2415c38da8e0a5e377ac17a82cd8856d25172a2608d2ca61053a8edfab4d236c",
    "closes": "b1354fe09b2f63d46e7129f2d1fafb868cc198e2c7881557258d204b6680f855",
    "dividends": "f46b52a85dae3bd49f1678b1cdf977440444a896d65d1d78ac69101e48620ba2",
}


def test_the_default_seed_is_the_one_the_golden_test_uses() -> None:
    assert world.DEFAULT_SEED == 1
    default, explicit = world.make_golden_data(), world.make_golden_data(1)
    pdt.assert_frame_equal(default.closes, explicit.closes)
    pdt.assert_frame_equal(default.opens, explicit.opens)


def test_prices_are_quantized_to_six_decimals() -> None:
    # last-bit exp/cumsum differences across numpy builds must not reach the store's bytes
    data = world.make_golden_data()
    for frame in (data.opens, data.closes):
        values = frame.to_numpy().ravel()
        values = values[~np.isnan(values)]
        assert (values == np.round(values, 6)).all()
        assert (values > 0).all()
    assert data.closes.loc["2023-02-01", "SPY"] == SAMPLE_SPY_FIRST_CLOSE
    assert data.opens.loc["2023-02-01", "AE1"] == SAMPLE_AE1_FIRST_OPEN


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
    # 2024-06-01 UTC): format 2, with the seed-1 macro series
    write_store(tmp_path / "store", world.make_golden_data(), created_at=world.FROZEN_AT)
    manifest = (tmp_path / "store" / MANIFEST).read_bytes()
    assert hashlib.sha256(manifest).hexdigest() == GOLDEN_STORE_DIGEST


def test_every_seed_keeps_the_events() -> None:
    data = world.make_golden_data(2)
    stopped = data.closes.index > pd.Timestamp(world.LAST_TRADED)
    assert data.closes.loc[stopped, world.DELIST_TICKER].isna().all()
    assert data.closes.loc[~stopped, world.DELIST_TICKER].notna().all()
    assert {span.ticker for span in data.membership} == set(world.SECTORS)
    assert data.splits == (Split(world.SPLIT_TICKER, world.SPLIT_EX, Fraction(2)),)
    assert data.dividends == (Dividend(world.DIVIDEND_TICKER, world.DIVIDEND_EX, world.DIVIDEND),)
    for ticker, filed, deficit in (
        (world.POISON_TICKER, world.LATE_FILED, -1_000_000.0),
        (world.BOUNDARY_TICKER, world.BOUNDARY_FILED, -2_000_000.0),
    ):
        rows = data.facts[
            (data.facts["ticker"] == ticker)
            & (data.facts["tag"] == world.POISONED_METRIC_TAG)
            & (data.facts["filed"] == filed)
        ]
        assert deficit in set(rows["value"])


def test_the_run_window_sits_inside_the_stores_price_dates() -> None:
    closes = world.make_golden_data().closes
    first, last = closes.index.min().date(), closes.index.max().date()

    assert first <= world.RUN_START <= world.LAST_DAY <= last
    assert first == world.FIRST_DAY
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


def _price_digests(data: FrozenData) -> dict[str, str]:
    def frame(values: pd.DataFrame) -> str:
        body = (
            values.to_numpy(dtype="float64").tobytes()
            + pd.DatetimeIndex(values.index).as_unit("ns").asi8.tobytes()
            + "|".join(map(str, values.columns)).encode("utf-8")
        )
        return hashlib.sha256(body).hexdigest()

    return {
        "opens": frame(data.opens),
        "closes": frame(data.closes),
        "dividends": hashlib.sha256(repr(data.dividends).encode("utf-8")).hexdigest(),
    }


def test_seed_one_opens_closes_and_dividends_are_byte_identical_to_before_macro() -> None:
    assert _price_digests(world.make_golden_data()) == SEED_ONE_PRICE_DIGESTS


def test_the_demo_macro_is_seeded_quantized_and_vix_moves_against_the_market() -> None:
    data = world.make_golden_data()
    macro = data.macro
    assert list(macro.columns) == list(MACRO_SERIES)
    assert macro.index.equals(data.closes.index)
    values = macro.to_numpy().ravel()
    assert (values == np.round(values, 6)).all()
    market = np.log(data.closes["SPY"]).diff()
    assert macro["VIXCLS"].diff().corr(market) < -0.5
    pdt.assert_frame_equal(world.make_golden_data(7).macro, world.make_golden_data(7).macro)
    assert not world.make_golden_data(2).macro.equals(macro)
