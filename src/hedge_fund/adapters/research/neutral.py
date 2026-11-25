"""The backtest's research stand-in: agrees with every shortlist side, at no cost."""

from __future__ import annotations

from datetime import date

from hedge_fund.domain.research import Rating, RunRecord
from hedge_fund.domain.shortlist import Shortlist, Side

_RATINGS = {Side.LONG: Rating.BUY, Side.SHORT: Rating.SELL}


class NeutralResearcher:
    """BUY for each shortlisted long, SELL for each short: every name is CONFIRM.

    Built per rebalance from the shortlist ``run_cycle`` will build (``plan_shortlist``).
    A name or date that is not that shortlist's raises; the research service records the
    error, and the backtest loop aborts on the resulting shortlist mismatch.
    """

    def __init__(self, shortlist: Shortlist) -> None:
        self._as_of = shortlist.as_of
        self._sides = {candidate.ticker: candidate.side for candidate in shortlist.candidates}

    def run(self, ticker: str, as_of: date, run_index: int) -> RunRecord:
        if as_of != self._as_of:
            raise ValueError(f"{as_of}: shortlist is as of {self._as_of}")
        side = self._sides.get(ticker)
        if side is None:
            raise ValueError(f"{ticker}: not on the {as_of} shortlist")
        return RunRecord(
            ticker=ticker,
            as_of=as_of,
            run_index=run_index,
            rating=_RATINGS[side],
            decision=f"neutral:{side.value}",
            reports={},
            seconds=0.0,
            error=None,
            usage=(),
        )
