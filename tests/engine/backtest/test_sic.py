"""The coarse SEC SIC code to GICS sector fallback."""

from __future__ import annotations

import pytest

from hedge_fund.engine.backtest.sic import SIC_RANGES, sic_sector
from hedge_fund.universe import GICS_SECTORS


@pytest.mark.parametrize(
    ("sic", "sector"),
    [
        ("2834", "Health Care"),
        ("2844", "Consumer Staples"),
        ("2860", "Materials"),
        ("1311", "Energy"),
        ("3674", "Information Technology"),
        ("3711", "Consumer Discretionary"),
        ("3721", "Industrials"),
        ("3841", "Health Care"),
        ("4911", "Utilities"),
        ("4813", "Communication Services"),
        ("5331", "Consumer Discretionary"),
        ("5411", "Consumer Staples"),
        ("5912", "Consumer Staples"),
        ("6021", "Financials"),
        ("6798", "Real Estate"),
        ("6512", "Real Estate"),
        ("7372", "Information Technology"),
        ("8062", "Health Care"),
        (" 7389 ", "Industrials"),
    ],
)
def test_codes_map_to_their_sector(sic: str, sector: str) -> None:
    assert sic_sector(sic) == sector


@pytest.mark.parametrize("sic", [None, "", "abc", "1100", "9995"])
def test_missing_malformed_or_unmapped_codes_give_none(sic: str | None) -> None:
    assert sic_sector(sic) is None


def test_every_range_is_ordered_and_names_a_gics_sector() -> None:
    for low, high, sector in SIC_RANGES:
        assert low <= high
        assert sector in GICS_SECTORS
