"""Tests for the per-ticker parquet cache decorator."""

import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from hedge_fund.adapters.data.caching import CachingProvider
from tests.factories import make_closes
from tests.fakes import FakeProvider, assert_closes_equal

TODAY = date(2026, 10, 9)
JAN = date(2026, 1, 5)
JUN = date(2026, 6, 1)
JUL = date(2026, 7, 1)
SEP_END = date(2026, 9, 30)


def _fake(tickers: Sequence[str] = ("AAA", "BBB")) -> FakeProvider:
    return FakeProvider(make_closes(list(tickers)), dict.fromkeys(tickers, "Energy"))


def _cache(fake: FakeProvider, root: Path, today: date = TODAY) -> CachingProvider:
    return CachingProvider(fake, root, clock=lambda: today)


def _coverage(root: Path, ticker: str) -> Any:
    return json.loads((root / "prices" / f"{ticker}.json").read_text(encoding="utf-8"))


def _slice(frame: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    return frame.loc[pd.Timestamp(start) : pd.Timestamp(end)]


def test_cold_fetch_writes_parquet_and_coverage(tmp_path: Path) -> None:
    fake = _fake()
    result = _cache(fake, tmp_path).closes(["AAA", "BBB"], JUN, SEP_END)
    assert_closes_equal(result, _slice(fake.frame, JUN, SEP_END))
    assert fake.closes_calls == [(("AAA", "BBB"), JUN, SEP_END)]
    assert (tmp_path / "prices" / "AAA.parquet").is_file()
    assert _coverage(tmp_path, "AAA") == {"start": "2026-06-01", "end": "2026-09-30"}


def test_covered_request_makes_no_inner_call(tmp_path: Path) -> None:
    fake = _fake()
    cache = _cache(fake, tmp_path)
    cache.closes(["AAA", "BBB"], JAN, SEP_END)
    result = cache.closes(["BBB", "AAA"], JUN, date(2026, 8, 31))
    assert len(fake.closes_calls) == 1
    assert_closes_equal(result, _slice(fake.frame[["BBB", "AAA"]], JUN, date(2026, 8, 31)))


def test_forward_extension_refetches_union_range(tmp_path: Path) -> None:
    fake = _fake(["AAA"])
    cache = _cache(fake, tmp_path)
    cache.closes(["AAA"], JAN, JUN)
    cache.closes(["AAA"], JUL, SEP_END)
    assert fake.closes_calls[-1] == (("AAA",), JAN, SEP_END)
    assert _coverage(tmp_path, "AAA") == {"start": "2026-01-05", "end": "2026-09-30"}


def test_backward_extension_refetches_union_range(tmp_path: Path) -> None:
    fake = _fake(["AAA"])
    cache = _cache(fake, tmp_path)
    cache.closes(["AAA"], JUN, SEP_END)
    cache.closes(["AAA"], JAN, JUL)
    assert fake.closes_calls[-1] == (("AAA",), JAN, SEP_END)
    assert _coverage(tmp_path, "AAA") == {"start": "2026-01-05", "end": "2026-09-30"}


def test_restated_series_replaces_cache_without_seam(tmp_path: Path) -> None:
    fake = _fake(["AAA"])
    cache = _cache(fake, tmp_path)
    cache.closes(["AAA"], JAN, JUN)
    fake.frame = fake.frame * 0.5  # a dividend restates every earlier adjusted close
    result = cache.closes(["AAA"], JAN, SEP_END)
    assert_closes_equal(result, _slice(fake.frame, JAN, SEP_END))
    again = cache.closes(["AAA"], JAN, JUN)
    assert_closes_equal(again, _slice(fake.frame, JAN, JUN))
    assert len(fake.closes_calls) == 2


def test_coverage_end_is_capped_before_today(tmp_path: Path) -> None:
    fake = _fake(["AAA"])
    cache = _cache(fake, tmp_path, today=date(2026, 9, 15))
    cache.closes(["AAA"], JUN, SEP_END)
    assert _coverage(tmp_path, "AAA") == {"start": "2026-06-01", "end": "2026-09-14"}
    cache.closes(["AAA"], JUN, SEP_END)
    assert len(fake.closes_calls) == 2


def test_request_for_today_only_writes_nothing(tmp_path: Path) -> None:
    fake = _fake(["AAA"])
    cache = _cache(fake, tmp_path, today=SEP_END)
    result = cache.closes(["AAA"], SEP_END, SEP_END)
    assert len(result) == 1
    assert not (tmp_path / "prices").exists()


def test_empty_fetch_is_not_cached_and_is_retried(tmp_path: Path) -> None:
    fake = _fake(["AAA"])
    cache = _cache(fake, tmp_path)
    result = cache.closes(["ZZZ"], JUN, SEP_END)
    assert list(result.columns) == ["ZZZ"]
    assert result.empty
    cache.closes(["ZZZ"], JUN, SEP_END)
    assert len(fake.closes_calls) == 2
    assert not (tmp_path / "prices").exists()


def test_empty_refetch_keeps_existing_cache(tmp_path: Path) -> None:
    fake = _fake(["AAA"])
    cache = _cache(fake, tmp_path)
    cache.closes(["AAA"], JUN, SEP_END)
    original = fake.frame
    fake.frame = original.drop(columns=["AAA"])  # vendor now returns nothing for AAA
    result = cache.closes(["AAA"], JAN, SEP_END)
    assert_closes_equal(result, _slice(original, JUN, SEP_END))
    assert _coverage(tmp_path, "AAA") == {"start": "2026-06-01", "end": "2026-09-30"}


def test_uncovered_tickers_with_same_range_share_one_fetch(tmp_path: Path) -> None:
    fake = _fake(["AAA", "BBB", "CCC"])
    cache = _cache(fake, tmp_path)
    cache.closes(["AAA"], JUN, SEP_END)
    result = cache.closes(["AAA", "BBB", "CCC"], JUN, SEP_END)
    assert fake.closes_calls == [(("AAA",), JUN, SEP_END), (("BBB", "CCC"), JUN, SEP_END)]
    assert_closes_equal(result, _slice(fake.frame, JUN, SEP_END))


def test_new_instance_reuses_cache(tmp_path: Path) -> None:
    fake = _fake()
    _cache(fake, tmp_path).closes(["AAA", "BBB"], JUN, SEP_END)
    result = _cache(fake, tmp_path).closes(["AAA", "BBB"], JUN, SEP_END)
    assert len(fake.closes_calls) == 1
    assert_closes_equal(result, _slice(fake.frame, JUN, SEP_END))


def test_parquet_without_coverage_is_treated_as_uncached(tmp_path: Path) -> None:
    fake = _fake(["AAA"])
    cache = _cache(fake, tmp_path)
    cache.closes(["AAA"], JUN, SEP_END)
    (tmp_path / "prices" / "AAA.json").unlink()
    cache.closes(["AAA"], JUN, SEP_END)
    assert len(fake.closes_calls) == 2


def test_corrupt_parquet_is_refetched_and_rewritten(tmp_path: Path) -> None:
    fake = _fake(["AAA"])
    cache = _cache(fake, tmp_path)
    cache.closes(["AAA"], JUN, SEP_END)
    parquet = tmp_path / "prices" / "AAA.parquet"
    parquet.write_bytes(b"not a parquet file")
    result = cache.closes(["AAA"], JUN, SEP_END)
    assert len(fake.closes_calls) == 2
    assert_closes_equal(result, _slice(fake.frame, JUN, SEP_END))
    assert len(pd.read_parquet(parquet)) == len(result)


def test_corrupt_coverage_json_is_refetched_and_rewritten(tmp_path: Path) -> None:
    fake = _fake(["AAA"])
    cache = _cache(fake, tmp_path)
    cache.closes(["AAA"], JUN, SEP_END)
    (tmp_path / "prices" / "AAA.json").write_text("{not json", encoding="utf-8")
    result = cache.closes(["AAA"], JUN, SEP_END)
    assert len(fake.closes_calls) == 2
    assert_closes_equal(result, _slice(fake.frame, JUN, SEP_END))
    assert _coverage(tmp_path, "AAA") == {"start": "2026-06-01", "end": "2026-09-30"}


def test_coverage_json_missing_keys_is_refetched(tmp_path: Path) -> None:
    fake = _fake(["AAA"])
    cache = _cache(fake, tmp_path)
    cache.closes(["AAA"], JUN, SEP_END)
    (tmp_path / "prices" / "AAA.json").write_text("{}", encoding="utf-8")
    cache.closes(["AAA"], JUN, SEP_END)
    assert len(fake.closes_calls) == 2


def test_corrupt_sectors_file_is_rewritten(tmp_path: Path) -> None:
    fake = _fake(["AAA"])
    (tmp_path / "sectors.json").write_text("{not json", encoding="utf-8")
    assert _cache(fake, tmp_path).sectors(["AAA"]) == {"AAA": "Energy"}
    assert fake.sectors_calls == [("AAA",)]
    stored = json.loads((tmp_path / "sectors.json").read_text(encoding="utf-8"))
    assert stored == {"AAA": "Energy"}


def test_non_dict_sectors_file_is_rewritten(tmp_path: Path) -> None:
    fake = _fake(["AAA"])
    (tmp_path / "sectors.json").write_text('["AAA"]', encoding="utf-8")
    assert _cache(fake, tmp_path).sectors(["AAA"]) == {"AAA": "Energy"}
    assert fake.sectors_calls == [("AAA",)]
    stored = json.loads((tmp_path / "sectors.json").read_text(encoding="utf-8"))
    assert stored == {"AAA": "Energy"}


def test_duplicate_tickers_are_fetched_once(tmp_path: Path) -> None:
    fake = _fake(["AAA"])
    cache = _cache(fake, tmp_path)
    result = cache.closes(["AAA", "AAA"], JUN, SEP_END)
    assert fake.closes_calls == [(("AAA",), JUN, SEP_END)]
    assert list(result.columns) == ["AAA", "AAA"]
    cache.sectors(["AAA", "AAA"])
    assert fake.sectors_calls == [("AAA",)]


def test_column_missing_from_inner_frame_is_empty_and_not_cached(tmp_path: Path) -> None:
    fake = _fake(["AAA", "BBB"])
    fake.frame = fake.frame.drop(columns=["BBB"])
    inner_closes = fake.closes

    def without_column(tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        return inner_closes(tickers, start, end).drop(columns=["BBB"])

    fake.closes = without_column  # type: ignore[method-assign]
    result = _cache(fake, tmp_path).closes(["AAA", "BBB"], JUN, SEP_END)
    assert list(result.columns) == ["AAA", "BBB"]
    assert result["BBB"].isna().all()
    assert result["AAA"].notna().all()
    assert not (tmp_path / "prices" / "BBB.parquet").exists()


def test_unsafe_ticker_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not safe"):
        _cache(_fake(["AAA"]), tmp_path).closes(["../AAA"], JUN, SEP_END)


def test_sectors_are_cached_after_first_lookup(tmp_path: Path) -> None:
    fake = FakeProvider(make_closes(["AAA"]), {"AAA": "Energy", "BBB": "Financials"})
    first = _cache(fake, tmp_path).sectors(["AAA", "BBB"])
    second = _cache(fake, tmp_path).sectors(["BBB", "AAA"])
    assert first == {"AAA": "Energy", "BBB": "Financials"}
    assert list(second.items()) == [("BBB", "Financials"), ("AAA", "Energy")]
    assert fake.sectors_calls == [("AAA", "BBB")]


def test_unknown_sectors_are_not_cached_and_are_retried(tmp_path: Path) -> None:
    fake = FakeProvider(make_closes(["AAA"]), {"AAA": "Energy"})
    cache = _cache(fake, tmp_path)
    assert cache.sectors(["AAA", "ZZZ"]) == {"AAA": "Energy", "ZZZ": None}
    assert cache.sectors(["AAA", "ZZZ"]) == {"AAA": "Energy", "ZZZ": None}
    assert fake.sectors_calls == [("AAA", "ZZZ"), ("ZZZ",)]
    stored = json.loads((tmp_path / "sectors.json").read_text(encoding="utf-8"))
    assert stored == {"AAA": "Energy"}
