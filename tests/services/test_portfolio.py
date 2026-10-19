"""Tests for the portfolio service: benchmark fetch and end-to-end portfolio build."""

from collections.abc import Sequence
from datetime import date, timedelta

import pandas as pd
import pytest

from hedge_fund.services.portfolio import fetch_benchmark
from tests.fakes import FakeProvider

AS_OF = date(2026, 9, 30)  # Wednesday
NAN = float("nan")


class _LeakyProvider(FakeProvider):
    """Ignores `end` and returns every row it has, including rows after as_of."""

    def closes(self, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        return self.frame.reindex(columns=list(tickers))


def _spy_frame(values: list[float], end: str = "2026-09-30") -> pd.DataFrame:
    index = pd.bdate_range(end=end, periods=len(values))
    return pd.DataFrame({"SPY": values, "QQQ": [1.0] * len(values)}, index=index)


# --- fetch_benchmark -------------------------------------------------------


def test_fetch_benchmark_requests_lookback_plus_one_rows_of_calendar() -> None:
    # 253 trading rows -> ceil(253 * 365 / 252) + 30 = 367 + 30 = 397 calendar days.
    fake = FakeProvider(_spy_frame([400.0, NAN, 402.0]), {})
    out = fetch_benchmark(fake, AS_OF, lookback_days=252)
    assert fake.closes_calls == [(("SPY",), AS_OF - timedelta(days=397), AS_OF)]
    assert out.tolist() == [400.0, 402.0]  # the NaN row is dropped
    assert out.index[-1] == pd.Timestamp(AS_OF)


def test_fetch_benchmark_uses_the_given_ticker() -> None:
    fake = FakeProvider(_spy_frame([400.0, 401.0]), {})
    out = fetch_benchmark(fake, AS_OF, lookback_days=10, ticker="QQQ")
    assert fake.closes_calls[0][0] == ("QQQ",)
    assert out.tolist() == [1.0, 1.0]


def test_fetch_benchmark_drops_rows_after_as_of() -> None:
    fake = _LeakyProvider(_spy_frame([400.0, 401.0, 402.0], end="2026-10-02"), {})
    out = fetch_benchmark(fake, AS_OF, lookback_days=10)
    assert out.tolist() == [400.0]  # 2026-10-01 and 2026-10-02 are after as_of


def test_fetch_benchmark_rejects_a_missing_ticker() -> None:
    # The provider returns an all-NaN column for an unknown ticker, not an empty frame.
    fake = FakeProvider(_spy_frame([400.0, 401.0]), {})
    with pytest.raises(ValueError, match="no 'XYZ' closes"):
        fetch_benchmark(fake, AS_OF, lookback_days=10, ticker="XYZ")
