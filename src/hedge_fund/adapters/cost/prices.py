"""The user's LLM price table as TOML: USD per million tokens per model; no built-in rates.

Format::

    fallback_run_cost = 0.8          # optional: USD per research run without history

    [models."claude-opus-5-5"]
    input = 15.0
    output = 75.0
    cache_write = 18.75
    cache_read = 1.5
"""

from __future__ import annotations

import math
import tomllib
from pathlib import Path
from typing import Any

from hedge_fund.domain.cost import ModelPrice, PriceTable

PRICES_FILE = "prices.toml"  # default name, under the data root
_TOP_KEYS = frozenset({"fallback_run_cost", "models"})
_RATE_KEYS = ("input", "output", "cache_write", "cache_read")


def load_prices(path: Path) -> PriceTable:
    """The table at ``path``; an empty table without a fallback when the file is missing.

    Raises ValueError, naming the file, for malformed TOML, an unknown top-level key, a
    model table without exactly the four rates, or a rate or ``fallback_run_cost`` that is
    not a finite number >= 0.
    """
    path = Path(path)
    if not path.exists():
        return PriceTable()
    try:
        body = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{path}: malformed price file: {exc}") from exc
    unknown = sorted(set(body) - _TOP_KEYS)
    if unknown:
        raise ValueError(f"{path}: unknown keys {unknown}")
    models = body.get("models", {})
    if not isinstance(models, dict):
        raise ValueError(f"{path}: models must be a table")
    fallback = body.get("fallback_run_cost")
    return PriceTable(
        models={name: _model_price(path, name, rates) for name, rates in models.items()},
        fallback_run_cost=None if fallback is None else _rate(path, "fallback_run_cost", fallback),
    )


def _model_price(path: Path, name: str, rates: Any) -> ModelPrice:
    label = f"models.{name!r}"
    if not isinstance(rates, dict):
        raise ValueError(f"{path}: {label} must be a table")
    if set(rates) != set(_RATE_KEYS):
        raise ValueError(
            f"{path}: {label} needs exactly {', '.join(_RATE_KEYS)}, got {', '.join(sorted(rates))}"
        )
    return ModelPrice(**{key: _rate(path, f"{label}.{key}", rates[key]) for key in _RATE_KEYS})


def _rate(path: Path, label: str, value: Any) -> float:
    """Type check first: bool and NaN would slip past a plain ``value >= 0``."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{path}: {label} must be a finite number >= 0, got {value!r}")
    return float(value)
