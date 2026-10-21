"""Order generation parameters, validated like the Layer 5 breaker params."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class OrderParams:
    """Leave a held name alone while |target weight - current weight| < no_trade_band."""

    no_trade_band: float = 0.0025


DEFAULT_ORDERS = OrderParams()


def validate_order_params(params: OrderParams) -> None:
    """Type check first: bool and NaN would slip past plain range comparisons."""
    names = ("no_trade_band",)
    for name in names:
        value = getattr(params, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
    for name in names:
        value = getattr(params, name)
        if not 0 <= value < 1:
            raise ValueError(f"{name} must satisfy 0 <= {name} < 1, got {value!r}")
