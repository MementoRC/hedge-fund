"""Approval proposal: the plain-text summary a human approves, and the plan fingerprint.

Pure: no I/O, imports only the domain and ``engine.execution``. ``services/approval.py``
shows the summary through an ApprovalGate.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence

from hedge_fund.domain.orders import BLOCKED_HALT, BLOCKED_VETO, Order, OrderAction, OrderPlan
from hedge_fund.domain.risk import BreakerState, RiskBreakdown, RiskVerdict, Violation
from hedge_fund.engine.execution import PreparedSubmission, prepare_submission

NA = "n/a"
NONE = "(none)"
DEFERRED_NOTE = "   deferred: side flip"
TOP_RESIDUALS = 5
TOP_FACTORS = 3
TOP_NAMES = 5

_RISK_LABELS: dict[str | None, str] = {
    None: "APPROVED",
    BLOCKED_HALT: "HALT",
    BLOCKED_VETO: "VETO",
}
_HEADER = ("leg", "ticker", "action", "shares", "ref", "notional")
_LEFT_ALIGNED = 3  # leg, ticker and action; shares, ref and notional are right-aligned
_LABEL_WIDTH = 14
_BUYS = frozenset({OrderAction.BUY, OrderAction.BUY_TO_COVER})


def render_proposal(verdict: RiskVerdict, plan: OrderPlan) -> str:
    """Plain-ASCII summary of ``plan`` and ``verdict``, ending with a newline.

    Blocks, separated by one blank line: header, orders, skipped and largest residuals,
    risk, violations. Leg labels come from ``prepare_submission``, so they are exactly what
    ``execute_plan`` will do. Raises ValueError when ``prepare_submission`` rejects the plan
    or ``plan.blocked`` is not None, BLOCKED_HALT or BLOCKED_VETO.
    """
    prepared = prepare_submission(plan)
    blocks = (
        [f"Proposal for {plan.as_of.isoformat()}   risk: {_risk_label(plan.blocked)}"],
        _orders(prepared),
        [
            "Skipped   " + (", ".join(f"{t} {r}" for t, r in plan.skipped.items()) or NONE),
            "Largest residuals   " + _top(plan.residual_weight, TOP_RESIDUALS),
        ],
        _risk(verdict.breakdown, verdict.breakers),
        _violations(verdict.violations),
    )
    return "\n\n".join("\n".join(block) for block in blocks) + "\n"


def plan_fingerprint(plan: OrderPlan) -> str:
    """sha256 hex digest of ``<as_of>|<ticker>,<action>,<shares>;...`` in plan order.

    It covers exactly the fields that build the 6a client_order_ids, so two plans with the
    same fingerprint submit the same orders. Prices, notionals, skips, residuals and
    ``blocked`` do not change it.
    """
    legs = ";".join(f"{order.ticker},{order.action.value},{order.shares}" for order in plan.orders)
    return hashlib.sha256(f"{plan.as_of.isoformat()}|{legs}".encode()).hexdigest()


def _risk_label(blocked: str | None) -> str:
    label = _RISK_LABELS.get(blocked)
    if label is None:
        raise ValueError(f"unknown blocked value {blocked!r}")
    return label


def _orders(prepared: PreparedSubmission) -> list[str]:
    """The orders table in plan order (reducing first), then the totals line."""
    legs = [("reduce", request) for request in prepared.reducing]
    legs += [("open", request) for request in prepared.opening]
    if not legs:
        return [f"Orders   {NONE}"]
    rows = [_HEADER] + [_row(leg, request.order) for leg, request in legs]
    notes = [""] + [
        DEFERRED_NOTE if request.client_order_id in prepared.deferred else "" for _, request in legs
    ]
    widths = [max(len(row[column]) for row in rows) for column in range(len(_HEADER))]
    table = [_aligned(row, widths) + note for row, note in zip(rows, notes, strict=True)]
    totals = _totals([request.order for _, request in legs])
    return ["Orders (reducing first)", *table, "  ---", totals]


def _row(leg: str, order: Order) -> tuple[str, ...]:
    return (
        leg,
        order.ticker,
        order.action.value,
        str(order.shares),
        _money(order.ref_price),
        _money(order.notional),
    )


def _aligned(row: Sequence[str], widths: Sequence[int]) -> str:
    cells = [
        cell.ljust(width) if column < _LEFT_ALIGNED else cell.rjust(width)
        for column, (cell, width) in enumerate(zip(row, widths, strict=True))
    ]
    return "  " + "  ".join(cells)


def _totals(orders: Sequence[Order]) -> str:
    """Deferred legs count: they are part of the plan being approved."""
    buys = sum(order.notional for order in orders if order.action in _BUYS)
    sells = sum(order.notional for order in orders if order.action not in _BUYS)
    count = "1 order" if len(orders) == 1 else f"{len(orders)} orders"
    return (
        f"  {count}   buys {_money(buys)}   sells {_money(sells)}"
        f"   gross {_money(buys + sells)}   net {_money(buys - sells, signed=True)}"
    )


def _risk(breakdown: RiskBreakdown, breakers: BreakerState) -> list[str]:
    """Vol line, top factors (shares of factor variance), top names, unassessed, breakers."""
    factor_variance = breakdown.factor_vol**2
    factor_shares = (
        {name: value / factor_variance for name, value in breakdown.by_factor.items()}
        if factor_variance != 0
        else {}
    )
    lines = [
        f"Risk   vol {_pct(breakdown.total_vol)}"
        f"  factor share {_pct(breakdown.factor_share)}"
        f"  specific {_pct(breakdown.specific_vol)}",
        _labelled("top factors", _top(factor_shares, TOP_FACTORS)),
        _labelled("top names", _top(breakdown.by_name, TOP_NAMES)),
    ]
    if breakdown.unassessed:
        lines.append(_labelled("unassessed", ", ".join(breakdown.unassessed)))
    daily = _pct(breakers.daily_loss, signed=True) + _tripped(breakers.daily_tripped)
    drawdown = _pct(breakers.drawdown) + _tripped(breakers.drawdown_tripped)
    lines.append(_labelled("breakers", f"daily {daily}   drawdown {drawdown}"))
    return lines


def _labelled(label: str, text: str) -> str:
    return f"  {label.ljust(_LABEL_WIDTH)}{text}"


def _tripped(tripped: bool) -> str:
    return " TRIPPED" if tripped else ""


def _violations(violations: Sequence[Violation]) -> list[str]:
    if not violations:
        return [f"Violations   {NONE}"]
    lines = ["Violations"]
    for violation in violations:
        line = (
            f"  {violation.check}  observed {_num(violation.observed)}"
            f"  limit {_num(violation.limit)}"
        )
        lines.append(f"{line}  {violation.detail}" if violation.detail else line)
    return lines


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
