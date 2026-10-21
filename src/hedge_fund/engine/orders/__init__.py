"""Order generation: target portfolio -> whole-share orders, gated by the risk verdict."""

from hedge_fund.engine.orders.generate import generate_orders
from hedge_fund.engine.orders.params import DEFAULT_ORDERS, OrderParams, validate_order_params

__all__ = [
    "DEFAULT_ORDERS",
    "OrderParams",
    "generate_orders",
    "validate_order_params",
]
