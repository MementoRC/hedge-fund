"""Approval proposal: the plain-text summary a human approves, and the plan fingerprint.

Pure: no I/O, imports only the domain and ``engine.execution``. ``services/approval.py``
shows the summary through an ApprovalGate.
"""

from __future__ import annotations

import hashlib

from hedge_fund.domain.orders import OrderPlan


def plan_fingerprint(plan: OrderPlan) -> str:
    """sha256 hex digest of ``<as_of>|<ticker>,<action>,<shares>;...`` in plan order.

    It covers exactly the fields that build the 6a client_order_ids, so two plans with the
    same fingerprint submit the same orders. Prices, notionals, skips, residuals and
    ``blocked`` do not change it.
    """
    legs = ";".join(f"{order.ticker},{order.action.value},{order.shares}" for order in plan.orders)
    return hashlib.sha256(f"{plan.as_of.isoformat()}|{legs}".encode()).hexdigest()
