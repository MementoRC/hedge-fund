"""Tests for build_snapshot: provider data to a validated PriceSnapshot."""

from collections.abc import Mapping, Sequence
from datetime import date, timedelta

import pandas as pd
import pytest

from hedge_fund.engine.factors.score import score
from hedge_fund.services.snapshots import build_snapshot
from tests.factories import make_closes
from tests.fakes import FakeProvider

AS_OF = date(2026, 9, 30)
SECTORS: dict[str, str | None] = {
    "AAA": "Energy",
    "BBB": "Energy",
    "CCC": "Energy",
    "DDD": "Financials",
    "EEE": "Financials",
    "FFF": "Financials",
}


def _fake(sectors: Mapping[str, str | None] = SECTORS) -> FakeProvider:
    return FakeProvider(make_closes(list(SECTORS), n_days=400), sectors)


class _LeakyProvider(FakeProvider):
    """Ignores `end` and returns every row it has, including rows after as_of."""

    def closes(self, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        return self.frame.reindex(columns=list(tickers))


def test_tickers_are_normalized_and_deduplicated() -> None:
    fake = _fake()
    build = build_snapshot(fake, [" aaa", "BBB", "aaa", "bbb "], AS_OF)
    assert list(build.snapshot.closes.columns) == ["AAA", "BBB"]
    assert fake.closes_calls[0][0] == ("AAA", "BBB")
    assert fake.sectors_calls == [("AAA", "BBB")]


def test_default_lookback_requests_465_calendar_days_ending_at_as_of() -> None:
    fake = _fake()
    build_snapshot(fake, ["AAA"], AS_OF)
    assert fake.closes_calls[0][1:] == (AS_OF - timedelta(days=465), AS_OF)


def test_larger_lookback_widens_the_requested_window() -> None:
    fake = _fake()
    build_snapshot(fake, ["AAA"], AS_OF, lookback=600)
    assert fake.closes_calls[0][1:] == (AS_OF - timedelta(days=900), AS_OF)


def test_rows_after_as_of_are_dropped() -> None:
    leaky = _LeakyProvider(make_closes(list(SECTORS)), SECTORS)
    build = build_snapshot(leaky, ["AAA"], date(2026, 9, 15))
    assert build.snapshot.closes.index[-1] == pd.Timestamp("2026-09-15")


def test_keeps_last_lookback_rows() -> None:
    build = build_snapshot(_fake(), ["AAA"], AS_OF, lookback=50)
    assert len(build.snapshot.closes) == 50
    assert build.snapshot.closes.index[-1] == pd.Timestamp(AS_OF)


def test_default_lookback_is_300_rows() -> None:
    assert len(build_snapshot(_fake(), ["AAA"], AS_OF).snapshot.closes) == 300


def test_missing_sector_drops_ticker_into_unresolved() -> None:
    build = build_snapshot(_fake({**SECTORS, "BBB": None}), ["CCC", "ZZZ", "BBB", "AAA"], AS_OF)
    assert list(build.snapshot.closes.columns) == ["CCC", "AAA"]
    assert dict(build.snapshot.sectors) == {"CCC": "Energy", "AAA": "Energy"}
    assert list(build.unresolved.items()) == [("BBB", "no_sector"), ("ZZZ", "no_sector")]


def test_ticker_without_prices_stays_as_nan_column() -> None:
    build = build_snapshot(_fake({**SECTORS, "NEW": "Energy"}), ["AAA", "NEW"], AS_OF)
    assert build.snapshot.closes["NEW"].isna().all()
    assert dict(build.unresolved) == {"NEW": "no_prices"}


def test_no_sector_takes_precedence_over_no_prices() -> None:
    build = build_snapshot(_fake(), ["AAA", "ZZZ"], AS_OF)
    assert dict(build.unresolved) == {"ZZZ": "no_sector"}


def test_empty_ticker_list_raises() -> None:
    with pytest.raises(ValueError, match="no tickers"):
        build_snapshot(_fake(), [" ", ""], AS_OF)


def test_all_tickers_unresolved_raises() -> None:
    with pytest.raises(ValueError, match="no price data"):
        build_snapshot(_fake({}), ["AAA"], AS_OF)


def test_no_rows_at_as_of_raises() -> None:
    with pytest.raises(ValueError, match="no price data"):
        build_snapshot(_fake(), ["AAA"], date(2020, 1, 1))


def test_end_to_end_build_then_score_ranks_tickers() -> None:
    build = build_snapshot(_fake({**SECTORS, "NEW": "Energy"}), [*SECTORS, "NEW"], AS_OF)
    result = score(build.snapshot)
    assert result.as_of == AS_OF
    assert dict(build.unresolved) == {"NEW": "no_prices"}
    assert {s.ticker for s in result.scores} == set(SECTORS)
    assert dict(result.excluded) == {"NEW": "insufficient_history"}
