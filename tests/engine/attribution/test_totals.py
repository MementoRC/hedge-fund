"""aggregate: totals by ticker, by sector (Unknown is a bucket) and overall."""

from __future__ import annotations

from datetime import date

from hedge_fund.domain.attribution import NO_PNL, AttributionTotals, DayAttribution
from hedge_fund.engine.attribution.totals import aggregate

TUE, WED = date(2026, 9, 29), date(2026, 9, 30)
ROWS = (
    DayAttribution(TUE, "AAA", "Tech", 10, 15.0, 5.0, 0.5, 2.0, True),
    DayAttribution(TUE, "BBB", "Unknown", -4, 0.0, 0.0, 4.0, 0.0, False),
    DayAttribution(WED, "AAA", "Tech", 10, -3.0, 1.0, -8.0, -1.5, True),
    DayAttribution(WED, "CCC", "Tech", 3, 1.0, 0.25, 0.75, 0.0, True),
    DayAttribution(WED, "GAP", "Energy", 5, 2.0, 2.0, 2.0, 2.0, True),
)


def test_totals_by_ticker_sector_and_overall_leave_out_excluded_tickers() -> None:
    by_ticker, by_sector, total = aggregate(ROWS, {"GAP"})
    assert by_ticker == {
        "AAA": AttributionTotals(12.0, 6.0, -7.5, 0.5, 11.0),
        "BBB": AttributionTotals(0.0, 0.0, 4.0, 0.0, 4.0),
        "CCC": AttributionTotals(1.0, 0.25, 0.75, 0.0, 2.0),
    }
    assert by_sector == {
        "Tech": AttributionTotals(13.0, 6.25, -6.75, 0.5, 13.0),
        "Unknown": AttributionTotals(0.0, 0.0, 4.0, 0.0, 4.0),
    }
    assert total == AttributionTotals(13.0, 6.25, -2.75, 0.5, 17.0)
    assert (list(by_ticker), list(by_sector)) == (["AAA", "BBB", "CCC"], ["Tech", "Unknown"])


def test_nothing_gives_zero() -> None:
    assert aggregate((), ()) == ({}, {}, NO_PNL)


def test_every_ticker_excluded_gives_zero() -> None:
    assert aggregate(ROWS, {"AAA", "BBB", "CCC", "GAP"}) == ({}, {}, NO_PNL)


def test_package_exports() -> None:
    import hedge_fund.engine.attribution as attribution

    assert attribution.__all__ == [
        "DayModel",
        "Trade",
        "aggregate",
        "attribute_day",
        "day_model",
        "holdings",
        "trades",
        "trading_date",
        "trading_days",
    ]
    assert attribution.aggregate is aggregate
