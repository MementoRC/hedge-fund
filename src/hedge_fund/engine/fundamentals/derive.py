"""Point-in-time fundamentals derivation over the long facts frame (ports/fundamentals.py).

Pure functions, no I/O: PIT filter, version selection, duration classes, reference periods,
TTM flows, stock values, shares and staleness. Dates follow the facts frame: datetime64 with
NaT `start` for instant facts.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from hedge_fund.engine.fundamentals.metrics import CURATED_TAGS, REPORT_FORMS

_VERSION_KEY = ["ticker", "tag", "start", "end"]


def pit_filter(facts: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Curated-tag facts from 10-K/10-Q forms (or amendments) filed strictly before `as_of`.

    EDGAR filing dates are calendar dates: a filing made after the close on `as_of` must not
    reach a snapshot priced at that close, so `filed == as_of` is excluded.
    """
    keep = (
        (facts["filed"] < pd.Timestamp(as_of))
        & facts["tag"].isin(CURATED_TAGS)
        & facts["form"].isin(REPORT_FORMS)
    )
    return facts.loc[keep]


def select_versions(facts: pd.DataFrame) -> pd.DataFrame:
    """Most recently filed fact per (ticker, tag, start, end): the latest public restatement.

    NaT `start` (instant facts) is a valid group key (dropna=False). Same-day versions are
    ordered by accession number. The result is sorted by tag, end and start.
    """
    ordered = facts.sort_values(["filed", "accn"], kind="stable")
    latest = ordered.groupby(_VERSION_KEY, dropna=False, sort=False).tail(1)
    return latest.sort_values(["tag", "end", "start"], kind="stable")
