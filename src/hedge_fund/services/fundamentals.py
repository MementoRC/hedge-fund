"""Build validated FundamentalsSnapshots from a FundamentalsProvider."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from hedge_fund.domain.fundamentals import FundamentalsSnapshot
from hedge_fund.engine.fundamentals.derive import derive_fundamentals
from hedge_fund.ports.fundamentals import FundamentalsProvider, coerce_facts, empty_facts
from hedge_fund.services.snapshots import normalize_tickers

NO_CIK = "no_cik"


@dataclass(frozen=True)
class FundamentalsBuild:
    """A fundamentals snapshot plus the tickers dropped while building it (ticker -> reason)."""

    snapshot: FundamentalsSnapshot
    unresolved: Mapping[str, str]


def build_fundamentals_snapshot(
    provider: FundamentalsProvider,
    tickers: Sequence[str],
    as_of: date,
    *,
    ciks: Mapping[str, str],
) -> FundamentalsBuild:
    """Fetch facts for `tickers` and derive a snapshot from filings with `filed < as_of`.

    Tickers absent from `ciks` are "no_cik" and are not requested. Others may be
    "no_filings", "no_reference_period" or "stale_fundamentals" (see engine derive).
    Raises ValueError when no ticker resolves.
    """
    requested = normalize_tickers(tickers)
    if not requested:
        raise ValueError("no tickers requested")
    with_cik = [ticker for ticker in requested if ticker in ciks]
    unresolved = {ticker: NO_CIK for ticker in requested if ticker not in ciks}
    facts = coerce_facts(provider.facts(with_cik)) if with_cik else empty_facts()
    derived = derive_fundamentals(facts, with_cik, as_of)
    unresolved.update(derived.unresolved)
    if derived.current.empty:
        raise ValueError("no fundamentals for the requested tickers at as_of")
    snapshot = FundamentalsSnapshot(
        as_of=as_of, current=derived.current, prior=derived.prior, periods=derived.periods
    )
    return FundamentalsBuild(snapshot=snapshot, unresolved=dict(sorted(unresolved.items())))
