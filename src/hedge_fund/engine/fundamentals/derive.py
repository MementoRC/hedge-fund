"""Point-in-time fundamentals derivation over the long facts frame (ports/fundamentals.py).

Pure functions, no I/O: PIT filter, version selection, duration classes, reference periods,
TTM flows, stock values, shares and staleness. Dates follow the facts frame: datetime64 with
NaT `start` for instant facts.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

import numpy as np
import pandas as pd

from hedge_fund.engine.fundamentals.metrics import CURATED_TAGS, REPORT_FORMS

_VERSION_KEY = ["ticker", "tag", "start", "end"]

QUARTER = "quarter"
HALF_YEAR = "half_year"
NINE_MONTH = "nine_month"
ANNUAL = "annual"
DURATION_CLASSES: Mapping[str, tuple[int, int]] = {
    QUARTER: (80, 100),
    HALF_YEAR: (170, 190),
    NINE_MONTH: (260, 280),
    ANNUAL: (350, 380),
}
TOLERANCE = pd.Timedelta(days=7)
ONE_YEAR = pd.DateOffset(years=1)


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
    return facts.loc[keep].reset_index(drop=True)


def select_versions(facts: pd.DataFrame) -> pd.DataFrame:
    """Most recently filed fact per (ticker, tag, start, end): the latest public restatement.

    NaT `start` (instant facts) is a valid group key (dropna=False). Same-day versions are
    ordered by accession number. The result is sorted by ticker, tag, end and start, with a
    fresh RangeIndex.
    """
    ordered = facts.sort_values(["filed", "accn"], kind="stable")
    latest = ordered.groupby(_VERSION_KEY, dropna=False, sort=False).tail(1)
    return latest.sort_values(["ticker", "tag", "end", "start"], kind="stable").reset_index(
        drop=True
    )


def duration_classes(facts: pd.DataFrame) -> pd.Series:
    """Duration class per fact by `end - start + 1` days; "" for instants and other lengths.

    Quarter 80-100, half-year YTD 170-190, nine-month YTD 260-280, annual 350-380 days
    (inclusive), so 52/53-week years (364/371 days) are annual.
    """
    days = (facts["end"] - facts["start"]).dt.days + 1
    conditions = [days.between(low, high).to_numpy() for low, high in DURATION_CLASSES.values()]
    labels = np.select(conditions, list(DURATION_CLASSES), default="")
    return pd.Series(labels, index=facts.index, dtype="str")


def reference_period(versioned: pd.DataFrame) -> pd.Timestamp | None:
    """E: the latest `end` among curated non-dei facts, or None when there is none."""
    ends = versioned.loc[_non_dei(versioned), "end"]
    return None if ends.empty else ends.max()


def prior_reference(versioned: pd.DataFrame, end: pd.Timestamp) -> pd.Timestamp:
    """E′: the curated non-dei fact end nearest `end - 1 year` within ±7 days, else NaT."""
    ends = versioned.loc[_non_dei(versioned), ["end"]].drop_duplicates().sort_values("end")
    hit = _nearest(ends, "end", end - ONE_YEAR)
    return pd.NaT if hit is None else hit["end"]


def last_filed(visible: pd.DataFrame) -> pd.Timestamp:
    """The latest `filed` among the PIT-filtered curated facts (dei included)."""
    return visible["filed"].max()


def _non_dei(facts: pd.DataFrame) -> pd.Series:
    return ~facts["tag"].str.startswith("dei:")


def _nearest(frame: pd.DataFrame, column: str, target: pd.Timestamp) -> pd.Series | None:
    """Row whose `column` is nearest `target` within TOLERANCE, or None.

    Equal distances keep frame order, so with frames sorted by date the earlier date wins.
    """
    distance = (frame[column] - target).abs()
    within = distance[distance <= TOLERANCE]
    if within.empty:
        return None
    row: pd.Series = frame.loc[within.sort_values(kind="stable").index[0]]
    return row
