"""Tests for the per-ticker parquet cache decorator for fundamentals."""

import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from hedge_fund.adapters.data import fundamentals_caching
from hedge_fund.adapters.data.fundamentals_caching import CachingFundamentalsProvider
from hedge_fund.engine.fundamentals.metrics import TAGS_HASH
from hedge_fund.ports.fundamentals import FACT_COLUMNS, FundamentalsProviderError, coerce_facts
from tests.factories import make_facts
from tests.fakes import FakeFundamentalsProvider

NOW = datetime(2026, 10, 11, 12, 0, tzinfo=UTC)
TTL = timedelta(days=1)


def _cache(
    fake: FakeFundamentalsProvider, root: Path, now: datetime = NOW
) -> CachingFundamentalsProvider:
    return CachingFundamentalsProvider(fake, root, clock=lambda: now, ttl=TTL)


def _meta(root: Path, ticker: str) -> Any:
    return json.loads((root / "fundamentals" / f"{ticker}.json").read_text(encoding="utf-8"))


def _rows(facts: pd.DataFrame, ticker: str) -> pd.DataFrame:
    return coerce_facts(facts.loc[facts["ticker"] == ticker])


def test_cold_fetch_writes_parquet_and_meta(tmp_path: Path) -> None:
    facts = make_facts(["AAA", "BBB"])
    fake = FakeFundamentalsProvider(facts)
    result = _cache(fake, tmp_path).facts(["AAA", "BBB"])
    pd.testing.assert_frame_equal(result, facts)
    assert fake.calls == [("AAA", "BBB")]
    assert (tmp_path / "fundamentals" / "AAA.parquet").is_file()
    assert _meta(tmp_path, "AAA") == {"fetched": NOW.isoformat(), "tags_hash": TAGS_HASH}


def test_fresh_entry_is_served_without_inner_call(tmp_path: Path) -> None:
    facts = make_facts(["AAA"])
    fake = FakeFundamentalsProvider(facts)
    _cache(fake, tmp_path).facts(["AAA"])
    result = _cache(fake, tmp_path, NOW + TTL - timedelta(seconds=1)).facts(["AAA"])
    assert len(fake.calls) == 1
    pd.testing.assert_frame_equal(result, facts)


def test_expired_entry_is_refetched_and_replaced(tmp_path: Path) -> None:
    facts = make_facts(["AAA"])
    fake = FakeFundamentalsProvider(facts)
    _cache(fake, tmp_path).facts(["AAA"])
    fake.frame = facts.assign(value=facts["value"] * 2.0)  # new filings since the last fetch
    later = NOW + TTL
    result = _cache(fake, tmp_path, later).facts(["AAA"])
    assert len(fake.calls) == 2
    pd.testing.assert_frame_equal(result, coerce_facts(fake.frame))
    assert _meta(tmp_path, "AAA")["fetched"] == later.isoformat()


def test_default_ttl_is_one_day(tmp_path: Path) -> None:
    fake = FakeFundamentalsProvider(make_facts(["AAA"]))
    for now in (NOW, NOW + timedelta(hours=23)):
        CachingFundamentalsProvider(fake, tmp_path, clock=lambda now=now: now).facts(["AAA"])
    assert len(fake.calls) == 1
    CachingFundamentalsProvider(fake, tmp_path, clock=lambda: NOW + timedelta(days=1)).facts(
        ["AAA"]
    )
    assert len(fake.calls) == 2


def test_tags_hash_mismatch_is_a_miss(tmp_path: Path) -> None:
    fake = FakeFundamentalsProvider(make_facts(["AAA"]))
    _cache(fake, tmp_path).facts(["AAA"])
    meta = tmp_path / "fundamentals" / "AAA.json"
    meta.write_text(json.dumps({"fetched": NOW.isoformat(), "tags_hash": "old"}), encoding="utf-8")
    _cache(fake, tmp_path).facts(["AAA"])
    assert len(fake.calls) == 2
    assert _meta(tmp_path, "AAA")["tags_hash"] == TAGS_HASH


@pytest.mark.parametrize("name", ["AAA.parquet", "AAA.json"])
def test_corrupt_entry_self_heals(tmp_path: Path, name: str) -> None:
    facts = make_facts(["AAA"])
    fake = FakeFundamentalsProvider(facts)
    _cache(fake, tmp_path).facts(["AAA"])
    (tmp_path / "fundamentals" / name).write_bytes(b"{not valid")
    healed = _cache(fake, tmp_path).facts(["AAA"])
    again = _cache(fake, tmp_path).facts(["AAA"])
    assert len(fake.calls) == 2
    pd.testing.assert_frame_equal(healed, facts)
    pd.testing.assert_frame_equal(again, facts)


@pytest.mark.parametrize("name", ["AAA.parquet", "AAA.json"])
def test_partial_entry_with_a_missing_file_self_heals(tmp_path: Path, name: str) -> None:
    facts = make_facts(["AAA"])
    fake = FakeFundamentalsProvider(facts)
    _cache(fake, tmp_path).facts(["AAA"])
    (tmp_path / "fundamentals" / name).unlink()
    healed = _cache(fake, tmp_path).facts(["AAA"])
    again = _cache(fake, tmp_path).facts(["AAA"])
    assert len(fake.calls) == 2
    pd.testing.assert_frame_equal(healed, facts)
    pd.testing.assert_frame_equal(again, facts)


def test_empty_result_is_not_cached_and_is_retried(tmp_path: Path) -> None:
    fake = FakeFundamentalsProvider(make_facts(["AAA"]))
    cache = _cache(fake, tmp_path)
    first = cache.facts(["ZZZ"])
    cache.facts(["ZZZ"])
    assert first.empty
    assert tuple(first.columns) == FACT_COLUMNS
    assert fake.calls == [("ZZZ",), ("ZZZ",)]
    assert not (tmp_path / "fundamentals").exists()


def test_requested_tickers_are_deduplicated(tmp_path: Path) -> None:
    facts = make_facts(["AAA", "BBB"])
    fake = FakeFundamentalsProvider(facts)
    result = _cache(fake, tmp_path).facts(["AAA", "BBB", "AAA"])
    assert fake.calls == [("AAA", "BBB")]
    pd.testing.assert_frame_equal(result, facts)


def test_only_misses_are_fetched_and_order_follows_the_request(tmp_path: Path) -> None:
    facts = make_facts(["AAA", "BBB"])
    fake = FakeFundamentalsProvider(facts)
    cache = _cache(fake, tmp_path)
    cache.facts(["AAA"])
    result = cache.facts(["BBB", "AAA"])
    assert fake.calls == [("AAA",), ("BBB",)]
    expected = pd.concat([_rows(facts, "BBB"), _rows(facts, "AAA")], ignore_index=True)
    pd.testing.assert_frame_equal(result, expected)


def test_unsafe_ticker_is_rejected_before_any_fetch(tmp_path: Path) -> None:
    fake = FakeFundamentalsProvider(make_facts(["AAA"]))
    with pytest.raises(ValueError, match="not safe"):
        _cache(fake, tmp_path).facts(["AAA", "../AAA"])
    assert fake.calls == []


def test_inner_error_propagates_and_nothing_is_cached(tmp_path: Path) -> None:
    class Failing:
        def facts(self, tickers: Sequence[str]) -> pd.DataFrame:
            raise FundamentalsProviderError("vendor down")

    cache = CachingFundamentalsProvider(Failing(), tmp_path, clock=lambda: NOW, ttl=TTL)
    with pytest.raises(FundamentalsProviderError, match="vendor down"):
        cache.facts(["AAA"])
    assert not (tmp_path / "fundamentals").exists()


def test_parquet_is_written_before_meta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    written: list[str] = []
    real = fundamentals_caching.atomic_write

    def spy(path: Path, writer: Callable[[Path], object]) -> None:
        written.append(path.name)
        real(path, writer)

    monkeypatch.setattr(fundamentals_caching, "atomic_write", spy)
    _cache(FakeFundamentalsProvider(make_facts(["AAA"])), tmp_path).facts(["AAA"])
    assert written == ["AAA.parquet", "AAA.json"]


def test_default_clock_is_utc_now(tmp_path: Path) -> None:
    before = datetime.now(UTC)
    CachingFundamentalsProvider(FakeFundamentalsProvider(make_facts(["AAA"])), tmp_path).facts(
        ["AAA"]
    )
    fetched = datetime.fromisoformat(_meta(tmp_path, "AAA")["fetched"])
    assert before <= fetched <= datetime.now(UTC)
