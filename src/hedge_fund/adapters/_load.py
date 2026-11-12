"""Read every ``*.json`` file of a report directory with one decoder."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hedge_fund.domain.loaded import Loaded


def load_dir[T](directory: Path, decode: Callable[[Any], T]) -> Loaded[T]:
    """Decode every ``directory/*.json`` in file-name order (temp files never match).

    A file that cannot be read or decoded is listed in ``unreadable`` under its name; a
    missing directory loads nothing.
    """
    items: list[T] = []
    names: list[str] = []
    unreadable: dict[str, str] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            items.append(decode(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, KeyError, TypeError, ValueError, AttributeError) as exc:
            unreadable[path.name] = f"{type(exc).__name__}: {exc}"
        else:
            names.append(path.name)
    return Loaded(items=tuple(items), unreadable=unreadable, names=tuple(names))
