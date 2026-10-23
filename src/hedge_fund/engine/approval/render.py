"""Approval proposal: the plain-text summary a human approves, and the plan fingerprint.

Pure: no I/O, imports only the domain and ``engine.execution``. ``services/approval.py``
shows the summary through an ApprovalGate.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping

from hedge_fund.domain.orders import OrderPlan

NA = "n/a"
NONE = "(none)"


def plan_fingerprint(plan: OrderPlan) -> str:
    """sha256 hex digest of ``<as_of>|<ticker>,<action>,<shares>;...`` in plan order.

    It covers exactly the fields that build the 6a client_order_ids, so two plans with the
    same fingerprint submit the same orders. Prices, notionals, skips, residuals and
    ``blocked`` do not change it.
    """
    legs = ";".join(f"{order.ticker},{order.action.value},{order.shares}" for order in plan.orders)
    return hashlib.sha256(f"{plan.as_of.isoformat()}|{legs}".encode()).hexdigest()


def _top(values: Mapping[str, float], limit: int) -> str:
    """Up to ``limit`` ``<key> <signed pct>`` entries, largest absolute value first.

    Ties go to the smaller key; ``(none)`` when ``values`` is empty.
    """
    ranked = sorted(values.items(), key=lambda item: (-abs(item[1]), item[0]))[:limit]
    return ", ".join(f"{key} {_pct(value, signed=True)}" for key, value in ranked) or NONE


def _money(value: float, *, signed: bool = False) -> str:
    """2 decimals with thousands separators (``9,244.00``)."""
    return _fixed(value, ",.2f", signed=signed)


def _pct(value: float, *, signed: bool = False) -> str:
    """``value`` x 100 with 2 decimals and a ``%`` (``11.80%``)."""
    text = _fixed(value * 100, ".2f", signed=signed)
    return text if text == NA else f"{text}%"


def _num(value: float) -> str:
    """Plain, 4 decimals (``0.1500``)."""
    return _fixed(value, ".4f", signed=False)


def _fixed(value: float, spec: str, *, signed: bool) -> str:
    """``value`` in format ``spec``, with a leading ``+``/``-`` when ``signed``.

    Non-finite values print as ``n/a``. A value that rounds to zero never prints as a
    negative zero: ``+0.00`` when signed, ``0.00`` when not.
    """
    if not math.isfinite(value):
        return NA
    sign = "+" if signed else ""
    text = format(value, sign + spec)
    if text.startswith("-") and not text.strip("-0.,"):
        return sign + text[1:]
    return text
