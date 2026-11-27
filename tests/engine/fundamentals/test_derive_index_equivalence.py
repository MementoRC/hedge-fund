"""The indexed TTM and stock lookups give what the original mask-and-sort scan gave.

The oracle below is the pre-index implementation, kept verbatim in behaviour, run over random
facts built on a coarse date grid so that ties, near misses and missing periods are common.
"""

import math
from typing import Any

import numpy as np
import pandas as pd

from hedge_fund.engine.fundamentals import derive
from hedge_fund.ports.fundamentals import FACT_COLUMNS, coerce_facts

_BASE = pd.Timestamp("2022-01-01")
_LENGTHS = [0, 90, 91, 89, 181, 182, 273, 274, 364, 365, 366, 200, 10]


def _nearest(frame: pd.DataFrame, column: str, target: pd.Timestamp) -> pd.Series | None:
    distance = (frame[column] - target).abs()
    mask = (distance <= derive.TOLERANCE).to_numpy()
    if not mask.any():
        return None
    candidates = frame.loc[mask].assign(_distance=distance.to_numpy()[mask])
    keys = ["_distance", column, *(name for name in frame.columns if name != column)]
    row: pd.Series = candidates.sort_values(keys, kind="stable").iloc[0].drop("_distance")
    return row


def _oracle_ytd(
    durations: pd.DataFrame, classes: pd.Series, annual: pd.DataFrame, end: pd.Timestamp
) -> tuple[pd.Series, str] | None:
    for label in (derive.NINE_MONTH, derive.HALF_YEAR, derive.QUARTER):
        pool = durations.loc[classes == label]
        if label == derive.QUARTER:
            opens = [
                _nearest(annual, "end", start - derive.ONE_DAY) is not None
                for start in pool["start"]
            ]
            pool = pool.loc[np.asarray(opens, dtype=bool)]
        hit = _nearest(pool, "end", end)
        if hit is not None:
            return hit, label
    return None


def _oracle_flow(facts: pd.DataFrame, end: pd.Timestamp) -> float:
    durations = facts.loc[facts["start"].notna()]
    classes = derive.duration_classes(durations)
    annual = durations.loc[classes == derive.ANNUAL]
    full_year = _nearest(annual, "end", end)
    if full_year is not None:
        return float(full_year["value"])
    found = _oracle_ytd(durations, classes, annual, end)
    if found is None:
        return math.nan
    ytd, label = found
    fy_prior = _nearest(annual, "end", ytd["start"] - derive.ONE_DAY)
    same = durations.loc[classes == label]
    near = (same["start"] - (ytd["start"] - derive.ONE_YEAR)).abs() <= derive.TOLERANCE
    ytd_prior = _nearest(same.loc[near], "end", end - derive.ONE_YEAR)
    if fy_prior is None or ytd_prior is None:
        return math.nan
    return float(ytd["value"] + fy_prior["value"] - ytd_prior["value"])


def _oracle_stock(facts: pd.DataFrame, end: pd.Timestamp) -> float:
    hit = _nearest(facts.loc[facts["start"].isna()], "end", end)
    return math.nan if hit is None else float(hit["value"])


def _random_facts(rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _ in range(int(rng.integers(0, 26))):
        step = 1 if rng.random() < 0.3 else 30
        jitter = int(rng.integers(-3, 4)) if rng.random() < 0.5 else 0
        end = _BASE + pd.Timedelta(days=int(rng.integers(0, 1100 // step)) * step + jitter)
        length = int(rng.choice(_LENGTHS))
        wobble = int(rng.integers(-5, 6)) if rng.random() < 0.5 else 0
        start = None if length == 0 else end - pd.Timedelta(days=length - 1 + wobble)
        value = math.nan if rng.random() < 0.1 else float(rng.integers(1, 500))
        rows.append(
            {
                "ticker": "AAA",
                "tag": "us-gaap:Revenues",
                "start": start,
                "end": None if rng.random() < 0.05 else end,
                "value": value,
                "filed": "2024-01-01",
                "form": "10-K",
                "accn": f"a{int(rng.integers(0, 6))}",
            }
        )
    facts = coerce_facts(pd.DataFrame(rows, columns=list(FACT_COLUMNS)))
    return facts.sample(frac=1.0, random_state=int(rng.integers(0, 99))).reset_index(drop=True)


def _same(left: float, right: float) -> bool:
    return (math.isnan(left) and math.isnan(right)) or left == right


def test_indexed_lookups_match_the_mask_and_sort_scan() -> None:
    rng = np.random.default_rng(7)
    for _ in range(60):
        facts = _random_facts(rng)
        for _ in range(12):
            end = _BASE + pd.Timedelta(days=int(rng.integers(0, 1100)))
            if rng.random() < 0.05:
                end = pd.NaT
            assert _same(derive.flow_ttm(facts, end), _oracle_flow(facts, end))
            assert _same(derive.stock_value(facts, end), _oracle_stock(facts, end))
