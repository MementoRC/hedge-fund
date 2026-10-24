"""Order generation parameters."""

from __future__ import annotations

import math

import pytest

from hedge_fund.engine.orders.params import DEFAULT_ORDERS, OrderParams, validate_order_params


def test_default_band() -> None:
    assert OrderParams(no_trade_band=0.0025) == DEFAULT_ORDERS
    validate_order_params(DEFAULT_ORDERS)


@pytest.mark.parametrize("band", [0, 0.0, 0.5, 0.999])
def test_accepts_band_in_range(band: float) -> None:
    validate_order_params(OrderParams(no_trade_band=band))


@pytest.mark.parametrize("band", [math.nan, math.inf, -math.inf, True, False, "0.1", None])
def test_rejects_non_finite_or_non_number_before_range(band: object) -> None:
    with pytest.raises(ValueError, match=r"no_trade_band must be a finite number, got "):
        validate_order_params(OrderParams(no_trade_band=band))  # type: ignore[arg-type]


@pytest.mark.parametrize("band", [-0.0001, -1, 1, 1.0, 2.5])
def test_rejects_band_out_of_range(band: float) -> None:
    with pytest.raises(
        ValueError, match=r"no_trade_band must satisfy 0 <= no_trade_band < 1, got "
    ):
        validate_order_params(OrderParams(no_trade_band=band))


def test_package_exports() -> None:
    import hedge_fund.engine.orders as orders

    assert orders.__all__ == [
        "DEFAULT_ORDERS",
        "OrderParams",
        "generate_orders",
        "order_prices",
        "validate_order_params",
    ]
