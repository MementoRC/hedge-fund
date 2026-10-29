"""attribute_day: holding P&L split by the day's model, the trading term, missing closes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date

import pandas as pd
import pytest

from hedge_fund.domain.attribution import DayAttribution
from hedge_fund.engine.attribution.book import Trade
from hedge_fund.engine.attribution.model import DayModel
from hedge_fund.engine.attribution.pnl import attribute_day

NAN = float("nan")
MON, TUE, WED, THU = date(2026, 9, 28), date(2026, 9, 29), date(2026, 9, 30), date(2026, 10, 1)
CLOSES = pd.DataFrame(
    {
        "AAA": [100.0, 102.0, 101.0, 104.0, 103.0, 105.0],
        "BBB": [50.0, 49.0, 51.0, 50.0, 52.0, 53.0],
        "NEW": [NAN, NAN, NAN, 20.0, 21.0, 22.0],
        "GAP": [30.0, 31.0, NAN, 32.0, 33.0, 34.0],
    },
    index=pd.to_datetime(
        ["2026-09-28", "2026-09-29", "2026-09-30", "2026-10-01", "2026-10-02", "2026-10-05"]
    ),
)
MODEL = DayModel(
    market=0.01,
    sector_returns={"Tech": 0.004, "Energy": -0.002},
    betas={"AAA": 1.5},
    sectors={"AAA": "Tech"},
    residuals={"AAA": 0.0},
)


def _row(
    day: date,
    ticker: str,
    sector: str = "Unknown",
    *,
    prior: int = 0,
    residual: float = 0.0,
    trading: float = 0.0,
) -> DayAttribution:
    """An unregressed row: no market or sector P&L."""
    return DayAttribution(
        day=day,
        ticker=ticker,
        sector=sector,
        prior_shares=prior,
        market=0.0,
        sector_pnl=0.0,
        residual=residual,
        trading=trading,
        regressed=False,
    )


def test_long_and_short_holdings_are_all_residual_without_a_model() -> None:
    rows, excluded = attribute_day(TUE, {"AAA": 10, "BBB": -4}, (), CLOSES, None, {})
    assert rows == (
        _row(TUE, "AAA", prior=10, residual=20.0),
        _row(TUE, "BBB", prior=-4, residual=4.0),
    )
    assert excluded == {}


def test_several_fills_in_a_day_give_the_trading_term() -> None:
    fills = (Trade("AAA", WED, 5, 100.5), Trade("AAA", WED, -3, 101.5))
    rows, _ = attribute_day(WED, {"AAA": 10}, fills, CLOSES, None, {"AAA": "Tech"})
    # holding 10 * (101 - 102); trading 5 * (101 - 100.5) - 3 * (101 - 101.5)
    assert rows == (_row(WED, "AAA", "Tech", prior=10, residual=-10.0, trading=4.0),)


def test_a_flip_across_zero_keeps_the_prior_holding_and_trades_the_rest() -> None:
    rows, _ = attribute_day(THU, {"AAA": 10}, (Trade("AAA", THU, -15, 103.0),), CLOSES, None, {})
    assert rows == (_row(THU, "AAA", prior=10, residual=30.0, trading=-15.0),)


def test_a_name_first_traded_on_its_listing_day_needs_no_previous_close() -> None:
    rows, excluded = attribute_day(THU, {}, (Trade("NEW", THU, 100, 19.5),), CLOSES, None, {})
    assert rows == (_row(THU, "NEW", trading=50.0),)
    assert excluded == {}


def test_a_trade_needs_only_that_days_close() -> None:
    rows, _ = attribute_day(THU, {}, (Trade("GAP", THU, 2, 31.5),), CLOSES, None, {})
    assert rows == (_row(THU, "GAP", trading=1.0),)


@pytest.mark.parametrize(
    ("day", "held", "fills", "ticker", "reason"),
    [
        pytest.param(WED, {"GAP": 5}, (), "GAP", "no close on 2026-09-30", id="held-no-close"),
        pytest.param(
            THU, {"GAP": 5}, (), "GAP", "no previous close for 2026-10-01", id="held-no-previous"
        ),
        pytest.param(
            WED, {}, (Trade("GAP", WED, 1, 30.0),), "GAP", "no close on 2026-09-30", id="traded"
        ),
        pytest.param(
            TUE, {"UNPRICED": 3}, (), "UNPRICED", "no close on 2026-09-29", id="no-column"
        ),
        pytest.param(
            MON, {"AAA": 1}, (), "AAA", "no previous close for 2026-09-28", id="first-day"
        ),
    ],
)
def test_a_name_missing_a_required_close_is_excluded(
    day: date, held: Mapping[str, int], fills: Sequence[Trade], ticker: str, reason: str
) -> None:
    assert attribute_day(day, held, fills, CLOSES, None, {}) == ((), {ticker: reason})


def test_an_excluded_name_does_not_stop_the_others() -> None:
    rows, excluded = attribute_day(WED, {"AAA": 10, "GAP": 5}, (), CLOSES, None, {})
    assert rows == (_row(WED, "AAA", prior=10, residual=-10.0),)
    assert excluded == {"GAP": "no close on 2026-09-30"}


def test_a_regressed_holding_splits_into_market_sector_and_residual() -> None:
    sectors = {"AAA": "Tech", "BBB": None}
    rows, _ = attribute_day(TUE, {"AAA": 10, "BBB": -4}, (), CLOSES, MODEL, sectors)
    aaa, bbb = rows
    assert (aaa.market, aaa.sector_pnl, aaa.residual) == pytest.approx((15.0, 4.0, 1.0))
    assert (aaa.sector, aaa.prior_shares, aaa.trading, aaa.regressed) == ("Tech", 10, 0.0, True)
    assert aaa.market + aaa.sector_pnl + aaa.residual == pytest.approx(20.0)
    assert bbb == _row(TUE, "BBB", prior=-4, residual=4.0)  # not in the model, no sector


def test_zero_holdings_are_ignored() -> None:
    assert attribute_day(TUE, {"AAA": 0}, (), CLOSES, MODEL, {}) == ((), {})
