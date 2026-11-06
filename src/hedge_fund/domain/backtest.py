"""Plain data for the backtest's frozen store."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Dividend:
    """A cash dividend of ``amount`` per share, ex-dated ``ex_date``.

    ``amount`` is on the frozen store's split basis (every split up to the freeze date
    applied), the basis of the stored Open and Close.
    """

    ticker: str
    ex_date: date
    amount: float
