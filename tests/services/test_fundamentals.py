"""Tests for build_fundamentals_snapshot: provider facts to a validated FundamentalsSnapshot."""

from collections.abc import Sequence
from datetime import date

import pandas as pd
import pytest

from hedge_fund.ports.fundamentals import FundamentalsProviderError
from hedge_fund.services.fundamentals import build_fundamentals_snapshot
from tests.factories import make_facts
from tests.fakes import FakeFundamentalsProvider

AS_OF = date(2026, 9, 30)
CIKS = {
    "AAA": "0000000001",
    "BBB": "0000000002",
    "OLD": "0000000003",
    "LATE": "0000000004",
    "GONE": "0000000005",
}


def test_resolved_tickers_get_current_and_prior_metrics() -> None:
    fake = FakeFundamentalsProvider(make_facts(["AAA", "BBB"]))
    build = build_fundamentals_snapshot(fake, ["BBB", "AAA"], AS_OF, ciks=CIKS)
    snapshot = build.snapshot
    assert build.unresolved == {}
    assert snapshot.as_of == AS_OF
    assert list(snapshot.current.index) == ["AAA", "BBB"]
    assert snapshot.current.loc["AAA", "total_assets"] == 1100.0
    assert snapshot.prior.loc["AAA", "total_assets"] == 950.0
    assert snapshot.current.loc["AAA", "shares_outstanding"] == 50.0
    assert snapshot.periods.loc["AAA"].tolist() == [
        pd.Timestamp("2026-06-30"),
        pd.Timestamp("2025-06-30"),
        pd.Timestamp("2026-08-01"),
    ]


def test_tickers_are_normalized_and_deduplicated() -> None:
    fake = FakeFundamentalsProvider(make_facts(["AAA", "BBB"]))
    build_fundamentals_snapshot(fake, [" aaa", "AAA", "bbb "], AS_OF, ciks=CIKS)
    assert fake.calls == [("AAA", "BBB")]


def test_ticker_without_cik_is_no_cik_and_not_requested() -> None:
    fake = FakeFundamentalsProvider(make_facts(["AAA", "NOCIK"]))
    build = build_fundamentals_snapshot(fake, ["AAA", "NOCIK"], AS_OF, ciks=CIKS)
    assert build.unresolved == {"NOCIK": "no_cik"}
    assert fake.calls == [("AAA",)]


def test_each_exclusion_reason_is_reported() -> None:
    old = make_facts(["OLD"])
    old = old.loc[old["filed"] < pd.Timestamp("2026-03-01")]  # latest E 2025-12-31: stale
    late = make_facts(["LATE"]).assign(filed=pd.Timestamp("2026-10-05"))  # nothing visible
    facts = pd.concat([make_facts(["AAA"]), old, late], ignore_index=True)
    build = build_fundamentals_snapshot(
        FakeFundamentalsProvider(facts), ["AAA", "OLD", "LATE", "GONE", "NOCIK"], AS_OF, ciks=CIKS
    )
    assert list(build.unresolved.items()) == [
        ("GONE", "no_filings"),
        ("LATE", "no_reference_period"),
        ("NOCIK", "no_cik"),
        ("OLD", "stale_fundamentals"),
    ]
    assert list(build.snapshot.current.index) == ["AAA"]


def test_all_tickers_unresolved_raises() -> None:
    fake = FakeFundamentalsProvider(make_facts(["AAA"]))
    with pytest.raises(ValueError, match="no fundamentals"):
        build_fundamentals_snapshot(fake, ["GONE", "NOCIK"], AS_OF, ciks=CIKS)


def test_no_ticker_with_a_cik_skips_the_provider_and_raises() -> None:
    fake = FakeFundamentalsProvider(make_facts(["AAA"]))
    with pytest.raises(ValueError, match="no fundamentals"):
        build_fundamentals_snapshot(fake, ["NOCIK"], AS_OF, ciks=CIKS)
    assert fake.calls == []


def test_provider_error_propagates() -> None:
    class RaisingProvider:
        def facts(self, tickers: Sequence[str]) -> pd.DataFrame:
            raise FundamentalsProviderError("boom")

    with pytest.raises(FundamentalsProviderError, match="boom"):
        build_fundamentals_snapshot(RaisingProvider(), ["AAA"], AS_OF, ciks=CIKS)


def test_empty_ticker_list_raises() -> None:
    fake = FakeFundamentalsProvider(make_facts(["AAA"]))
    with pytest.raises(ValueError, match="no tickers"):
        build_fundamentals_snapshot(fake, [" ", ""], AS_OF, ciks=CIKS)
