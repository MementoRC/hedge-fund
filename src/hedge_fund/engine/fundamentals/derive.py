"""Point-in-time fundamentals derivation over the long facts frame (ports/fundamentals.py).

Pure functions, no I/O: PIT filter, version selection, duration classes, reference periods,
TTM flows, stock values, shares and staleness. Dates follow the facts frame: datetime64 with
NaT `start` for instant facts.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
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
ONE_DAY = pd.Timedelta(days=1)
_YTD_CLASSES = (NINE_MONTH, HALF_YEAR, QUARTER)  # longest first

ValueAt = Callable[[pd.DataFrame, pd.Timestamp], float]


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
    """E: the latest `end` among curated non-dei facts, or None when there is none.

    Precondition: `versioned` holds one ticker's facts after `pit_filter` and `select_versions`.
    """
    ends = versioned.loc[_non_dei(versioned), "end"]
    return None if ends.empty else ends.max()


def prior_reference(versioned: pd.DataFrame, end: pd.Timestamp) -> pd.Timestamp:
    """E′: the curated non-dei fact end nearest `end - 1 year` within ±7 days.

    Precondition: `versioned` holds one ticker's facts after `pit_filter` and `select_versions`.
    Returns NaT when no end is within ±7 days (including an empty frame).
    """
    ends = versioned.loc[_non_dei(versioned), ["end"]].drop_duplicates().sort_values("end")
    hit = _nearest(ends, "end", end - ONE_YEAR)
    return pd.NaT if hit is None else hit["end"]


def last_filed(visible: pd.DataFrame) -> pd.Timestamp:
    """The latest `filed` among the PIT-filtered curated facts (dei included).

    Precondition: `visible` holds one ticker's facts after `pit_filter`, before version
    selection. Returns NaT on an empty frame.
    """
    return visible["filed"].max()


def flow_ttm(facts: pd.DataFrame, end: pd.Timestamp) -> float:
    """Trailing-twelve-month value at `end` from ONE tag's versioned facts; NaN if incomplete.

    An annual-class fact ending at `end` (±7 days) is the TTM. Otherwise
    TTM = YTD(end) + FY(prior) - YTD(end - 1 year): YTD(end) is the longest quarter,
    half-year or nine-month fact ending at `end` (a quarter only when an annual fact ends at
    its start - 1 day), FY(prior) the annual fact ending at YTD(end).start - 1 day, and
    YTD(end - 1 year) the same-class fact ending at `end - 1 year` whose start is within ±7
    days of YTD(end).start - 1 year. Nearest `end` wins among matches.
    """
    durations = facts.loc[facts["start"].notna()]
    classes = duration_classes(durations)
    annual = durations.loc[classes == ANNUAL]
    full_year = _nearest(annual, "end", end)
    if full_year is not None:
        return float(full_year["value"])
    found = _ytd(durations, classes, annual, end)
    if found is None:
        return math.nan
    ytd, ytd_class = found
    fy_prior = _nearest(annual, "end", ytd["start"] - ONE_DAY)
    same_class = durations.loc[classes == ytd_class]
    starts_match = (same_class["start"] - (ytd["start"] - ONE_YEAR)).abs() <= TOLERANCE
    ytd_prior = _nearest(same_class.loc[starts_match], "end", end - ONE_YEAR)
    if fy_prior is None or ytd_prior is None:
        return math.nan
    return float(ytd["value"] + fy_prior["value"] - ytd_prior["value"])


def first_complete(
    versioned: pd.DataFrame, tags: Sequence[str], end: pd.Timestamp, value_at: ValueAt
) -> float:
    """Value from the first tag, in order, that yields a complete value at `end`, else NaN.

    `value_at` sees one tag's facts only, so the pieces of one value never mix tags.
    """
    for tag in tags:
        value = value_at(versioned.loc[versioned["tag"] == tag], end)
        if not math.isnan(value):
            return value
    return math.nan


def _ytd(
    durations: pd.DataFrame, classes: pd.Series, annual: pd.DataFrame, end: pd.Timestamp
) -> tuple[pd.Series, str] | None:
    """Longest year-to-date fact ending at `end`; a quarter counts only right after a year."""
    for label in _YTD_CLASSES:
        pool = durations.loc[classes == label]
        if label == QUARTER:
            opens_year = [
                _nearest(annual, "end", start - ONE_DAY) is not None for start in pool["start"]
            ]
            pool = pool.loc[np.asarray(opens_year, dtype=bool)]
        hit = _nearest(pool, "end", end)
        if hit is not None:
            return hit, label
    return None


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
