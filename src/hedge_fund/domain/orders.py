"""Order types: signed holdings in, whole-share orders and an order plan out."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

SKIPPED_BAND = "band"
SKIPPED_ROUNDS_TO_ZERO = "rounds-to-zero"
BLOCKED_HALT = "halt"
BLOCKED_VETO = "veto"


class OrderAction(StrEnum):
    """Direction of one order; the order's ``shares`` is always positive."""

    BUY = "buy"
    SELL = "sell"
    SELL_SHORT = "sell_short"
    BUY_TO_COVER = "buy_to_cover"


@dataclass(frozen=True)
class Holding:
    """Current position; ``shares`` is signed (+ long, - short)."""

    ticker: str
    shares: int


@dataclass(frozen=True)
class Order:
    """One whole-share order; ``ref_price`` is the sizing price and ``notional = shares * ref_price``."""

    ticker: str
    action: OrderAction
    shares: int
    ref_price: float
    notional: float


@dataclass(frozen=True)
class OrderPlan:
    """Reducing legs then opening legs (each by ticker), plus skips and per-name residuals.

    ``skipped`` maps ticker -> SKIPPED_* and ``residual_weight`` maps ticker ->
    post-trade weight - target weight, both in ticker order. When ``blocked`` is set
    (BLOCKED_HALT or BLOCKED_VETO) ``orders`` is empty and residuals use current holdings.
    """

    as_of: date
    orders: tuple[Order, ...]
    skipped: Mapping[str, str]
    residual_weight: Mapping[str, float]
    blocked: str | None
