"""The dashboard's one chart palette."""

from __future__ import annotations

from hedge_fund.dashboard.palette import PALETTE
from hedge_fund.viz.theme import DARK


def test_the_dashboard_draws_with_one_fixed_dark_palette() -> None:
    assert PALETTE is DARK
