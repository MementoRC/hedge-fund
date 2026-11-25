"""The result of reading a directory of stored reports: what decoded and what did not."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Loaded[T]:
    """Every decodable item in file-name order, and the files that did not decode:
    file name -> ``"<ExceptionType>: <message>"``.
    """

    items: tuple[T, ...]
    unreadable: Mapping[str, str]
    names: tuple[str, ...] = ()
