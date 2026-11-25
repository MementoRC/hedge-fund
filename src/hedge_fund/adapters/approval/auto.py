"""An approval gate that approves every plan: used only by the backtest replay."""

from __future__ import annotations


class AutoApprovalGate:
    """Approve every summary. A risk veto never reaches a gate, so the veto still stands.

    Only the backtest composition root builds this gate; live cycles keep the console gate.
    """

    def confirm(self, summary: str) -> bool:
        return True
