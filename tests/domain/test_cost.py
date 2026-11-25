"""Tests for the LLM cost domain types."""

from __future__ import annotations

import dataclasses

import pytest

from hedge_fund.domain.cost import (
    BudgetError,
    BudgetPlan,
    Estimate,
    EstimateSource,
    ModelPrice,
    ModelUsage,
    PriceTable,
    ResearchRunError,
    Spend,
)

USAGE = ModelUsage(
    model="claude-opus-5-5",
    calls=2,
    input_tokens=1200,
    output_tokens=300,
    cache_write_tokens=100,
    cache_read_tokens=400,
)
PRICE = ModelPrice(input=15.0, output=75.0, cache_write=18.75, cache_read=1.5)
ESTIMATE = Estimate(cost=0.8, source=EstimateSource.HISTORY, n_samples=30)
PLAN = BudgetPlan(
    budget=80.0,
    estimate=ESTIMATE,
    requested_n=40,
    requested_k=3,
    n=40,
    k=2,
    projected=64.0,
    skipped=False,
    reasons=("k 3→2: 40×3×$0.80=$96.00 > $80.00",),
)
SPEND = Spend(total=1.5, runs=3, unknown_runs=1, usage=(USAGE,), over_budget=False)


def test_estimate_source_values() -> None:
    assert [source.value for source in EstimateSource] == ["history", "fallback"]


@pytest.mark.parametrize(
    ("cls", "names"),
    [
        pytest.param(
            ModelUsage,
            [
                "model",
                "calls",
                "input_tokens",
                "output_tokens",
                "cache_write_tokens",
                "cache_read_tokens",
            ],
            id="usage",
        ),
        pytest.param(ModelPrice, ["input", "output", "cache_write", "cache_read"], id="price"),
        pytest.param(PriceTable, ["models", "fallback_run_cost"], id="table"),
        pytest.param(Estimate, ["cost", "source", "n_samples"], id="estimate"),
        pytest.param(
            BudgetPlan,
            [
                "budget",
                "estimate",
                "requested_n",
                "requested_k",
                "n",
                "k",
                "projected",
                "skipped",
                "reasons",
            ],
            id="plan",
        ),
        pytest.param(Spend, ["total", "runs", "unknown_runs", "usage", "over_budget"], id="spend"),
    ],
)
def test_field_order(cls: type, names: list[str]) -> None:
    assert [field.name for field in dataclasses.fields(cls)] == names


def test_price_table_defaults_to_empty_without_a_fallback() -> None:
    table = PriceTable()
    assert table.models == {}
    assert table.fallback_run_cost is None
    assert PriceTable(models={"m": PRICE}, fallback_run_cost=0.8).models["m"] == PRICE


@pytest.mark.parametrize(
    ("instance", "field"),
    [
        pytest.param(USAGE, "calls", id="usage"),
        pytest.param(PRICE, "input", id="price"),
        pytest.param(PriceTable(), "fallback_run_cost", id="table"),
        pytest.param(ESTIMATE, "cost", id="estimate"),
        pytest.param(PLAN, "k", id="plan"),
        pytest.param(SPEND, "total", id="spend"),
    ],
)
def test_types_are_frozen(instance: object, field: str) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, field, None)


def test_budget_error_is_a_value_error() -> None:
    assert issubclass(BudgetError, ValueError)


def test_research_run_error_carries_its_message_and_usage() -> None:
    error = ResearchRunError("RuntimeError: boom", (USAGE,))
    assert str(error) == "RuntimeError: boom"
    assert error.usage == (USAGE,)
    assert ResearchRunError("x", None).usage is None
    assert isinstance(error, Exception)
