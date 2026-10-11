"""Tests for the FundamentalsProvider port and its facts-frame helpers."""

import pandas as pd
import pytest

from hedge_fund.ports.fundamentals import (
    FACT_COLUMNS,
    FundamentalsProvider,
    FundamentalsProviderError,
    coerce_facts,
    empty_facts,
)
from tests.factories import make_facts
from tests.fakes import FakeFundamentalsProvider


def test_fundamentals_provider_error_is_an_exception() -> None:
    assert issubclass(FundamentalsProviderError, Exception)


def test_fact_columns_are_the_port_contract() -> None:
    assert FACT_COLUMNS == ("ticker", "tag", "start", "end", "value", "filed", "form", "accn")


def test_empty_facts_has_canonical_columns_and_dtypes() -> None:
    facts = empty_facts()
    assert facts.empty
    assert tuple(facts.columns) == FACT_COLUMNS
    for column in ("start", "end", "filed"):
        assert facts[column].dtype == "datetime64[ns]"
    assert facts["value"].dtype == "float64"
    for column in ("ticker", "tag", "form", "accn"):
        assert pd.api.types.is_string_dtype(facts[column])


def test_coerce_facts_orders_columns_drops_extras_and_casts() -> None:
    raw = pd.DataFrame(
        {
            "accn": ["0001-24-000001"],
            "value": [5],
            "fy": [2024],
            "form": ["10-Q"],
            "filed": ["2024-05-01"],
            "end": ["2024-03-31"],
            "start": [None],
            "tag": ["us-gaap:Assets"],
            "ticker": ["AAA"],
        },
        index=[7],
    )
    facts = coerce_facts(raw)
    assert tuple(facts.columns) == FACT_COLUMNS
    assert facts.index.tolist() == [0]
    assert facts["value"].dtype == "float64"
    assert facts.loc[0, "value"] == 5.0
    assert pd.isna(facts.loc[0, "start"])
    assert facts.loc[0, "end"] == pd.Timestamp("2024-03-31")
    assert facts["filed"].dtype == "datetime64[ns]"


def _one_row() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA"],
            "tag": ["us-gaap:Assets"],
            "start": [None],
            "end": ["2024-03-31"],
            "value": [1.0],
            "filed": ["2024-05-01"],
            "form": ["10-Q"],
            "accn": ["0001-24-000001"],
        }
    )


@pytest.mark.parametrize("null", [None, float("nan")])
@pytest.mark.parametrize("column", ["ticker", "tag", "form", "accn"])
def test_coerce_facts_rejects_missing_text_values(column: str, null: object) -> None:
    raw = _one_row()
    raw[column] = [null]
    with pytest.raises(ValueError, match=column):
        coerce_facts(raw)


def test_coerce_facts_mixed_null_starts_keep_order_and_dtypes() -> None:
    raw = pd.concat([_one_row()] * 3, ignore_index=True)
    raw["ticker"] = ["A", "B", "C"]
    raw["start"] = [pd.NaT, "2024-01-01", None]
    facts = coerce_facts(raw)
    assert facts["ticker"].tolist() == ["A", "B", "C"]
    assert facts["start"].isna().tolist() == [True, False, True]
    assert facts["start"].dtype == "datetime64[ns]"
    assert facts["value"].dtype == "float64"


def test_coerce_facts_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="accn"):
        coerce_facts(empty_facts().drop(columns=["accn"]))


def test_fake_fundamentals_provider_satisfies_the_port() -> None:
    provider: FundamentalsProvider = FakeFundamentalsProvider(make_facts(["AAA", "BBB"]))
    facts = provider.facts(["BBB"])
    assert set(facts["ticker"]) == {"BBB"}
    assert tuple(facts.columns) == FACT_COLUMNS
