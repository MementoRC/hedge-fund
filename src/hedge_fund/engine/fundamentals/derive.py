"""Point-in-time fundamentals derivation over the long facts frame (ports/fundamentals.py).

Pure functions, no I/O: PIT filter, version selection, duration classes, reference periods,
TTM flows, stock values, shares and staleness. Dates follow the facts frame: datetime64 with
NaT `start` for instant facts.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from hedge_fund.domain.fundamentals import METRIC_NAMES, PERIOD_COLUMNS
from hedge_fund.engine.fundamentals.metrics import (
    CURATED_TAGS,
    METRICS,
    REPORT_FORMS,
    SHARES_DEI_TAG,
)

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

MAX_STALENESS_DAYS = 200
NO_FILINGS = "no_filings"
NO_REFERENCE_PERIOD = "no_reference_period"
STALE_FUNDAMENTALS = "stale_fundamentals"
_SHARES_METRIC = "shares_outstanding"


@dataclass(frozen=True)
class TickerFundamentals:
    """One resolved ticker: metrics at E and E′ plus (current_end, prior_end, last_filed)."""

    current: Mapping[str, float]
    prior: Mapping[str, float]
    periods: tuple[pd.Timestamp, pd.Timestamp | pd.NaTType, pd.Timestamp]


@dataclass(frozen=True)
class Derived:
    """Tickers x METRIC_NAMES frames at E and E′, tickers x PERIOD_COLUMNS dates, exclusions.

    Indexes are sorted tickers; `unresolved` maps ticker -> reason, sorted by ticker.
    """

    current: pd.DataFrame
    prior: pd.DataFrame
    periods: pd.DataFrame
    unresolved: Mapping[str, str]


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

    Precondition: `versioned` holds one ticker's facts after `pit_filter` and `select_versions`.
    `value_at` sees one tag's facts only, so the pieces of one value never mix tags.
    """
    for tag in tags:
        value = value_at(versioned.loc[versioned["tag"] == tag], end)
        if not math.isnan(value):
            return value
    return math.nan


def stock_value(facts: pd.DataFrame, end: pd.Timestamp) -> float:
    """Instant value nearest `end` (±7 days) from ONE tag's versioned facts, else NaN."""
    hit = _nearest(facts.loc[facts["start"].isna()], "end", end)
    return math.nan if hit is None else float(hit["value"])


def current_shares(visible: pd.DataFrame) -> float:
    """Cover-page share count from the most recently filed report that carries one, else NaN.

    Precondition: `visible` holds one ticker's facts after `pit_filter`. The filing is the dei
    fact with the greatest (`filed`, `end`, `accn`), so ties on `filed` go to the later cover
    date, then the greater accession, in any row order. See `_cover_count` for the value.
    """
    dei = visible.loc[visible["tag"] == SHARES_DEI_TAG]
    if dei.empty:
        return math.nan
    accn = dei.sort_values(["filed", "end", "accn"], kind="stable").iloc[-1]["accn"]
    return _cover_count(dei.loc[dei["accn"] == accn])


def prior_shares(visible: pd.DataFrame, prior_end: pd.Timestamp) -> float:
    """Cover-page share count of the filing that first reported `prior_end`, else NaN.

    The first reporter is the smallest (`filed`, `accn`) among non-dei facts ending exactly at
    `prior_end`. Precondition: `visible` is one ticker's PIT-filtered frame before version
    selection, which would otherwise discard that earlier filing's accession number. See
    `_cover_count` for the value.
    """
    reporting = visible.loc[(visible["end"] == prior_end) & _non_dei(visible)]
    if reporting.empty:
        return math.nan
    accn = reporting.sort_values(["filed", "accn"], kind="stable").iloc[0]["accn"]
    return _cover_count(visible.loc[(visible["tag"] == SHARES_DEI_TAG) & (visible["accn"] == accn)])


def _cover_count(dei: pd.DataFrame) -> float:
    """One filing's dei share count at its latest cover date; NaN if none or ambiguous.

    More than one distinct value at that date (e.g. per-class counts reported undimensioned)
    gives NaN rather than an arbitrary pick; identical duplicates are one value.
    """
    if dei.empty:
        return math.nan
    values = dei.loc[dei["end"] == dei["end"].max(), "value"].unique()
    return float(values[0]) if len(values) == 1 else math.nan


def metric_values(versioned: pd.DataFrame, end: pd.Timestamp) -> dict[str, float]:
    """Every metric at `end` by the tag rule (shares: only the us-gaap fallback tag here).

    Precondition: `versioned` holds one ticker's facts after `pit_filter` and `select_versions`.
    """
    values: dict[str, float] = {}
    for spec in METRICS:
        value_at: ValueAt = flow_ttm if spec.kind == "flow" else stock_value
        tags = [tag for tag in spec.tags if tag != SHARES_DEI_TAG]
        values[spec.name] = first_complete(versioned, tags, end, value_at)
    return values


def derive_ticker(facts: pd.DataFrame, as_of: date) -> TickerFundamentals | str:
    """One ticker's metrics at E and E′ as of `as_of`, or the reason it is unresolved.

    Flow and stock values come from the version-selected frame; dei shares and `last_filed`
    from the PIT-filtered frame before version selection. `E` older than
    `as_of - MAX_STALENESS_DAYS` is stale; exactly at the limit is kept. Without an E′ the
    prior metrics are all NaN.
    """
    visible = pit_filter(facts, as_of)
    versioned = select_versions(visible)
    end = reference_period(versioned)
    if end is None:
        return NO_REFERENCE_PERIOD
    if end < pd.Timestamp(as_of) - pd.Timedelta(days=MAX_STALENESS_DAYS):
        return STALE_FUNDAMENTALS
    prior_end = prior_reference(versioned, end)
    current = metric_values(versioned, end)
    _prefer_dei_shares(current, current_shares(visible))
    prior = dict.fromkeys(METRIC_NAMES, math.nan)
    if not pd.isna(prior_end):
        prior = metric_values(versioned, prior_end)
        _prefer_dei_shares(prior, prior_shares(visible, prior_end))
    return TickerFundamentals(
        current=current, prior=prior, periods=(end, prior_end, last_filed(visible))
    )


def derive_fundamentals(facts: pd.DataFrame, tickers: Sequence[str], as_of: date) -> Derived:
    """Derive every requested ticker from a multi-ticker facts frame.

    Tickers without rows are `no_filings`; see derive_ticker for the other reasons. Duplicate
    tickers count once.
    """
    by_ticker = dict(list(facts.reset_index(drop=True).groupby("ticker", sort=False)))
    resolved: dict[str, TickerFundamentals] = {}
    unresolved: dict[str, str] = {}
    for ticker in dict.fromkeys(tickers):
        group = by_ticker.get(ticker)
        result = NO_FILINGS if group is None else derive_ticker(group, as_of)
        if isinstance(result, str):
            unresolved[ticker] = result
        else:
            resolved[ticker] = result
    index = sorted(resolved)
    return Derived(
        current=_metric_frame([resolved[t].current for t in index], index),
        prior=_metric_frame([resolved[t].prior for t in index], index),
        periods=pd.DataFrame(
            [list(resolved[t].periods) for t in index], index=index, columns=list(PERIOD_COLUMNS)
        ).astype("datetime64[ns]"),
        unresolved=dict(sorted(unresolved.items())),
    )


def _prefer_dei_shares(values: dict[str, float], dei_shares: float) -> None:
    """The dei cover count wins; the us-gaap value stays only when no dei count qualifies."""
    if not math.isnan(dei_shares):
        values[_SHARES_METRIC] = dei_shares


def _metric_frame(rows: list[Mapping[str, float]], index: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [[row[name] for name in METRIC_NAMES] for row in rows],
        index=index,
        columns=list(METRIC_NAMES),
        dtype="float64",
    )


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

    Candidates are sorted by (distance, `column`, then the remaining columns), so an equal
    distance goes to the earlier date and the result never depends on input row order.
    """
    distance = (frame[column] - target).abs()
    mask = (distance <= TOLERANCE).to_numpy()
    if not mask.any():
        return None
    candidates = frame.loc[mask].assign(_distance=distance.to_numpy()[mask])
    keys = ["_distance", column, *(name for name in frame.columns if name != column)]
    row: pd.Series = candidates.sort_values(keys, kind="stable").iloc[0].drop("_distance")
    return row
