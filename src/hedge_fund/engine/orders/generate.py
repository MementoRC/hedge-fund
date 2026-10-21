"""Target portfolio -> whole-share orders, gated by the risk verdict.

Pure: prices and NAV come from the caller. Shares round toward zero, held names inside the
no-trade band are left alone, and a side flip is split into a close leg then an open leg.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from hedge_fund.domain.orders import (
    BLOCKED_HALT,
    BLOCKED_VETO,
    SKIPPED_BAND,
    SKIPPED_ROUNDS_TO_ZERO,
    Holding,
    Order,
    OrderAction,
    OrderPlan,
)
from hedge_fund.domain.portfolio import TargetPortfolio
from hedge_fund.domain.risk import RiskVerdict
from hedge_fund.engine.orders.params import DEFAULT_ORDERS, OrderParams, validate_order_params


def generate_orders(
    target: TargetPortfolio,
    verdict: RiskVerdict,
    holdings: tuple[Holding, ...],
    prices: Mapping[str, float],
    nav: float,
    *,
    params: OrderParams = DEFAULT_ORDERS,
) -> OrderPlan:
    """Orders moving ``holdings`` to ``target``, or none when the verdict blocks trading.

    Every input is validated before the verdict is consulted, so a halt never hides bad data.
    Raises ValueError on invalid params, mismatched dates, a non-finite or non-positive NAV or
    price, a non-finite weight, non-int shares, or a duplicate ticker.
    """
    validate_order_params(params)
    if verdict.as_of != target.as_of:
        raise ValueError(
            f"verdict as_of {verdict.as_of} does not match target as_of {target.as_of}"
        )
    _validate_nav(nav)
    weights = _target_weights(target)
    current = _current_shares(holdings)
    tickers = sorted(weights.keys() | current.keys())
    _validate_prices(tickers, prices)
    blocked = _blocked(verdict)
    if blocked is not None:
        drift = {
            ticker: current.get(ticker, 0) * prices[ticker] / nav - weights.get(ticker, 0.0)
            for ticker in tickers
        }
        return OrderPlan(
            as_of=target.as_of, orders=(), skipped={}, residual_weight=drift, blocked=blocked
        )
    orders: list[Order] = []
    skipped: dict[str, str] = {}
    residual: dict[str, float] = {}
    for ticker in tickers:
        price, target_w, held = prices[ticker], weights.get(ticker, 0.0), current.get(ticker, 0)
        final, reason = _final_shares(target_w, held, price, nav, params.no_trade_band)
        if reason is not None:
            skipped[ticker] = reason
        orders.extend(_legs(ticker, held, final, price))
        residual[ticker] = final * price / nav - target_w
    return OrderPlan(
        as_of=target.as_of,
        orders=tuple(orders),
        skipped=skipped,
        residual_weight=residual,
        blocked=None,
    )


def _final_shares(
    target_w: float, held: int, price: float, nav: float, band: float
) -> tuple[int, str | None]:
    """Shares after this cycle and the skip reason, if any; the first matching case wins."""
    if target_w == 0:
        return 0, None  # silent when not held, otherwise an exit that ignores the band
    desired = math.trunc(target_w * nav / price)
    if held == 0:
        return (0, SKIPPED_ROUNDS_TO_ZERO) if desired == 0 else (desired, None)
    if abs(target_w - held * price / nav) < band:
        return held, SKIPPED_BAND
    if desired == held:
        return held, SKIPPED_ROUNDS_TO_ZERO
    return desired, None


def _legs(ticker: str, held: int, final: int, price: float) -> list[Order]:
    """Orders moving ``held`` to ``final``; a side flip becomes a close leg then an open leg."""
    if final == held:
        return []
    if held > 0 > final:
        return [
            _order(ticker, OrderAction.SELL, held, price),
            _order(ticker, OrderAction.SELL_SHORT, -final, price),
        ]
    if held < 0 < final:
        return [
            _order(ticker, OrderAction.BUY_TO_COVER, -held, price),
            _order(ticker, OrderAction.BUY, final, price),
        ]
    if final > held:
        action = OrderAction.BUY_TO_COVER if held < 0 else OrderAction.BUY
        return [_order(ticker, action, final - held, price)]
    action = OrderAction.SELL if held > 0 else OrderAction.SELL_SHORT
    return [_order(ticker, action, held - final, price)]


def _order(ticker: str, action: OrderAction, shares: int, price: float) -> Order:
    return Order(
        ticker=ticker, action=action, shares=shares, ref_price=price, notional=shares * price
    )


def _blocked(verdict: RiskVerdict) -> str | None:
    """Halt outranks a plain veto."""
    if verdict.halt:
        return BLOCKED_HALT
    if not verdict.approved:
        return BLOCKED_VETO
    return None


def _is_real(value: object) -> bool:
    """A finite int or float; bool is excluded because it is an int subclass."""
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(value)


def _validate_nav(nav: float) -> None:
    if not _is_real(nav) or not nav > 0:
        raise ValueError(f"nav must be a finite number > 0, got {nav!r}")


def _target_weights(target: TargetPortfolio) -> dict[str, float]:
    weights: dict[str, float] = {}
    for position in target.positions:
        if not _is_real(position.weight):
            raise ValueError(
                f"weight must be a finite number, got {position.weight!r} for {position.ticker}"
            )
        if position.ticker in weights:
            raise ValueError(f"duplicate position ticker {position.ticker!r}")
        weights[position.ticker] = position.weight
    return weights


def _current_shares(holdings: tuple[Holding, ...]) -> dict[str, int]:
    shares: dict[str, int] = {}
    for holding in holdings:
        if isinstance(holding.shares, bool) or not isinstance(holding.shares, int):
            raise ValueError(f"shares must be an int, got {holding.shares!r} for {holding.ticker}")
        if holding.ticker in shares:
            raise ValueError(f"duplicate holding ticker {holding.ticker!r}")
        shares[holding.ticker] = holding.shares
    return shares


def _validate_prices(tickers: list[str], prices: Mapping[str, float]) -> None:
    """Every traded or held ticker needs a finite price > 0; extra entries are ignored."""
    for ticker in tickers:
        price = prices.get(ticker)
        if price is None or not _is_real(price) or not price > 0:
            raise ValueError(f"price for {ticker} must be a finite number > 0, got {price!r}")
