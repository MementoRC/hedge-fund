"""Live FRED smoke test. Excluded by default; run with: pytest -m network. No key needed."""

from datetime import date

import pytest

from hedge_fund.adapters.data.factory import default_macro_provider

pytestmark = pytest.mark.network


def test_fredgraph_serves_dgs10_for_a_past_month() -> None:
    frame = default_macro_provider().series(["DGS10"], date(2024, 1, 2), date(2024, 1, 31))
    assert list(frame.columns) == ["DGS10"]
    assert frame["DGS10"].notna().sum() >= 20  # January 2024 had 21 Treasury trading days
    assert 3.5 < frame["DGS10"].min() <= frame["DGS10"].max() < 4.5  # about 3.9 to 4.2
