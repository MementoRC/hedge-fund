"""The JSON shape shared by the report stores (cycle and settlement reports, fills)."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from datetime import date
from enum import Enum
from typing import Any


def jsonable(value: Any) -> Any:
    """Dataclasses as objects in field order, enums as values, dates and datetimes as ISO
    strings, mappings as objects, tuples and lists as lists, non-finite floats as
    ``"nan"`` / ``"inf"`` / ``"-inf"``.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]
    return value


_NON_FINITE = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}


def decode_float(value: Any) -> float:
    """Inverse of ``jsonable`` for one float field: a JSON number, or ``"nan"`` / ``"inf"`` /
    ``"-inf"``. ValueError for any other string, TypeError for a non-number (bools included).
    """
    if isinstance(value, str):
        if value not in _NON_FINITE:
            raise ValueError(f"{value!r} is not a float")
        return _NON_FINITE[value]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{value!r} is not a float")
    return float(value)
