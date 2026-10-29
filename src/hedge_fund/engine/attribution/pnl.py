"""One trading day's P&L per ticker: the holding P&L split by the day's model, plus the
trading P&L of the day's fills. Pure; no forward fill.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date

import pandas as pd

from hedge_fund.domain.attribution import UNKNOWN_SECTOR, DayAttribution
from hedge_fund.engine.attribution.book import Trade
from hedge_fund.engine.attribution.model import DayModel


def attribute_day(
    day: date,
    holdings_prev: Mapping[str, int],
    fills_today: Sequence[Trade],
    closes: pd.DataFrame,
    model: DayModel | None,
    sectors: Mapping[str, str | None],
) -> tuple[tuple[DayAttribution, ...], dict[str, str]]:
    """(rows sorted by ticker, excluded ticker -> reason) for ``day``.

    ``closes`` is indexed by the trading-day calendar and holds ``day``; the previous close
    is its row before ``day``'s. A ticker counts when held at the previous close
    (``holdings_prev``) or traded on ``day``. ``holding = h * (P_t - P_{t-1})`` and
    ``trading = sum q * (P_t - price)``. When ``model`` regressed the ticker, ``market =
    h P_{t-1} beta f_market``, ``sector_pnl = h P_{t-1} f_sector`` and ``residual`` is the
    rest of the holding P&L (``h P_{t-1} u`` up to rounding); otherwise the holding P&L is
    all residual. ``P_t`` is required for every counted ticker and ``P_{t-1}`` when held; a
    ticker missing one is excluded, with no row.
    """
    position = int(closes.index.get_loc(pd.Timestamp(day)))
    current = closes.iloc[position]
    previous = closes.iloc[position - 1] if position > 0 else None
    traded: dict[str, list[Trade]] = {}
    for trade in fills_today:
        traded.setdefault(trade.ticker, []).append(trade)
    held = {ticker for ticker, shares in holdings_prev.items() if shares}
    rows: list[DayAttribution] = []
    excluded: dict[str, str] = {}
    for ticker in sorted(held | traded.keys()):
        row = _row(
            day,
            ticker,
            holdings_prev.get(ticker, 0),
            traded.get(ticker, []),
            (_price(current, ticker), _price(previous, ticker)),
            model,
            sectors.get(ticker) or UNKNOWN_SECTOR,
        )
        if isinstance(row, str):
            excluded[ticker] = row
        else:
            rows.append(row)
    return tuple(rows), excluded


def _price(row: pd.Series | None, ticker: str) -> float | None:
    """The finite, positive close of ``ticker`` in ``row``, else None."""
    if row is None or ticker not in row.index:
        return None
    value = float(row[ticker])
    return value if math.isfinite(value) and value > 0 else None


def _row(
    day: date,
    ticker: str,
    prior: int,
    trades: Sequence[Trade],
    prices: tuple[float | None, float | None],
    model: DayModel | None,
    sector: str,
) -> DayAttribution | str:
    """The ticker's row, or the reason it is excluded."""
    close, previous = prices
    if close is None:
        return f"no close on {day}"
    trading = math.fsum(trade.shares * (close - trade.price) for trade in trades)
    if not prior:
        return _attribution(day, ticker, sector, 0, (0.0, 0.0, 0.0), trading, regressed=False)
    if previous is None:
        return f"no previous close for {day}"
    holding = prior * (close - previous)
    if model is None or ticker not in model.residuals:
        return _attribution(
            day, ticker, sector, prior, (0.0, 0.0, holding), trading, regressed=False
        )
    exposure = prior * previous
    market = exposure * model.betas[ticker] * model.market
    sector_pnl = exposure * model.sector_returns[model.sectors[ticker]]
    split = (market, sector_pnl, holding - market - sector_pnl)
    return _attribution(day, ticker, sector, prior, split, trading, regressed=True)


def _attribution(
    day: date,
    ticker: str,
    sector: str,
    prior: int,
    split: tuple[float, float, float],
    trading: float,
    *,
    regressed: bool,
) -> DayAttribution:
    market, sector_pnl, residual = split
    return DayAttribution(
        day=day,
        ticker=ticker,
        sector=sector,
        prior_shares=prior,
        market=market,
        sector_pnl=sector_pnl,
        residual=residual,
        trading=trading,
        regressed=regressed,
    )
