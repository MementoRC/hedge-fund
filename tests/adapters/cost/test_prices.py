"""Tests for the TOML LLM price table."""

from __future__ import annotations

from pathlib import Path

import pytest

from hedge_fund.adapters.cost import PRICES_FILE, load_prices
from hedge_fund.domain.cost import ModelPrice, PriceTable

VALID = """
fallback_run_cost = 0.8

[models."claude-opus-5-5"]
input = 15
output = 75.0
cache_write = 18.75
cache_read = 1.5

[models."claude-sonnet-5"]
input = 3.0
output = 15.0
cache_write = 3.75
cache_read = 0.3
"""
RATES = "input = 1.0\noutput = 1.0\ncache_write = 1.0\ncache_read = 1.0\n"


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / PRICES_FILE
    path.write_text(text, encoding="utf-8")
    return path


def test_the_default_file_name() -> None:
    assert PRICES_FILE == "prices.toml"


def test_a_valid_table_loads(tmp_path: Path) -> None:
    table = load_prices(_write(tmp_path, VALID))
    assert table == PriceTable(
        models={
            "claude-opus-5-5": ModelPrice(
                input=15.0, output=75.0, cache_write=18.75, cache_read=1.5
            ),
            "claude-sonnet-5": ModelPrice(input=3.0, output=15.0, cache_write=3.75, cache_read=0.3),
        },
        fallback_run_cost=0.8,
    )
    assert isinstance(table.models["claude-opus-5-5"].input, float)


def test_the_fallback_is_optional(tmp_path: Path) -> None:
    table = load_prices(_write(tmp_path, f'[models."m"]\n{RATES}'))
    assert table.fallback_run_cost is None
    assert set(table.models) == {"m"}


def test_only_a_fallback_is_fine(tmp_path: Path) -> None:
    assert load_prices(_write(tmp_path, "fallback_run_cost = 0\n")) == PriceTable(
        fallback_run_cost=0.0
    )


def test_a_missing_file_is_an_empty_table(tmp_path: Path) -> None:
    assert load_prices(tmp_path / PRICES_FILE) == PriceTable()


@pytest.mark.parametrize(
    ("text", "message"),
    [
        pytest.param("not = [toml", "malformed price file", id="malformed"),
        pytest.param("budget = 3\n", r"unknown keys \['budget'\]", id="unknown-top-key"),
        pytest.param("models = 3\n", "models must be a table", id="models-not-a-table"),
        pytest.param('[models]\n"m" = 3\n', "models.'m' must be a table", id="model-not-a-table"),
        pytest.param(
            f'[models."m"]\n{RATES}batch = 1.0\n', "models.'m' needs exactly", id="unknown-rate"
        ),
        pytest.param('[models."m"]\ninput = 1.0\n', "models.'m' needs exactly", id="missing-rate"),
        pytest.param(
            f'[models."m"]\n{RATES.replace("input = 1.0", "input = -1")}',
            r"models.'m'.input must be a finite number >= 0, got -1",
            id="negative",
        ),
        pytest.param(
            f'[models."m"]\n{RATES.replace("output = 1.0", "output = nan")}',
            r"models.'m'.output must be a finite number >= 0, got nan",
            id="nan",
        ),
        pytest.param(
            f'[models."m"]\n{RATES.replace("cache_read = 1.0", "cache_read = inf")}',
            "cache_read must be a finite number >= 0, got inf",
            id="inf",
        ),
        pytest.param(
            f'[models."m"]\n{RATES.replace("cache_write = 1.0", "cache_write = true")}',
            "cache_write must be a finite number >= 0, got True",
            id="bool",
        ),
        pytest.param(
            "fallback_run_cost = -0.5\n",
            "fallback_run_cost must be a finite number >= 0, got -0.5",
            id="negative-fallback",
        ),
        pytest.param(
            'fallback_run_cost = "cheap"\n',
            "fallback_run_cost must be a finite number >= 0, got 'cheap'",
            id="string-fallback",
        ),
    ],
)
def test_a_bad_table_is_a_value_error_naming_the_file(
    tmp_path: Path, text: str, message: str
) -> None:
    path = _write(tmp_path, text)
    with pytest.raises(ValueError, match=message) as info:
        load_prices(path)
    assert str(info.value).startswith(f"{path}: ")
