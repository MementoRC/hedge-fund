"""FundamentalsProvider port: raw long-format XBRL facts per ticker."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import pandas as pd

FACT_COLUMNS: tuple[str, ...] = ("ticker", "tag", "start", "end", "value", "filed", "form", "accn")
_DATE_COLUMNS = ("start", "end", "filed")
_TEXT_COLUMNS = ("ticker", "tag", "form", "accn")


class FundamentalsProviderError(Exception):
    """A vendor or transport failure while fetching fundamentals."""


class FundamentalsProvider(Protocol):
    """Source of reported XBRL facts. Implementations live in adapters/data/."""

    def facts(self, tickers: Sequence[str]) -> pd.DataFrame:
        """Long-format facts for the requested tickers, one row per reported fact.

        Columns are FACT_COLUMNS: `tag` is prefixed (us-gaap:..., dei:...), `start` is NaT
        for instant facts, `form` is a 10-K/10-Q form or amendment and `accn` the filing's
        accession number. A ticker with no facts contributes no rows. Raises
        FundamentalsProviderError on vendor or transport failure.
        """


def coerce_facts(frame: pd.DataFrame) -> pd.DataFrame:
    """`frame` restricted to FACT_COLUMNS in order, with canonical dtypes and a fresh index.

    Dates become datetime64[ns] (NaT allowed), `value` float64, text columns str.
    """
    missing = [column for column in FACT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"facts frame lacks columns: {missing}")
    facts = frame.loc[:, list(FACT_COLUMNS)].copy()
    for column in _DATE_COLUMNS:
        facts[column] = pd.to_datetime(facts[column]).astype("datetime64[ns]")
    facts["value"] = facts["value"].astype("float64")
    for column in _TEXT_COLUMNS:
        facts[column] = facts[column].astype("str")
    return facts.reset_index(drop=True)


def empty_facts() -> pd.DataFrame:
    """A facts frame with no rows and the canonical columns and dtypes."""
    return coerce_facts(pd.DataFrame(columns=list(FACT_COLUMNS)))
