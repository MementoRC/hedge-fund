"""Re-freeze the flagged spans of an existing store without fetching anything.

Services never import adapters: the CLI loads the source store, calls ``reflag`` and writes
the result with adapters/data/frozen_store.py.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

from hedge_fund.domain.backtest import FlaggedSpan, FrozenData
from hedge_fund.services.backtest_snapshot import frozen_flagged


@dataclass(frozen=True)
class Reflag:
    """``data`` with the new flagged spans, and how they differ from the old ones."""

    data: FrozenData
    added: tuple[FlaggedSpan, ...]
    removed: tuple[FlaggedSpan, ...]


def reflag(data: FrozenData, flagged: Sequence[FlaggedSpan]) -> Reflag:
    """``data`` with only its flagged spans replaced by those of ``flagged`` overlapping the
    store's window, the way ``build_frozen_data`` freezes them."""
    kept = frozen_flagged(flagged, data.start, data.end)
    return Reflag(
        data=dataclasses.replace(data, flagged=kept),
        added=tuple(item for item in kept if item not in data.flagged),
        removed=tuple(item for item in data.flagged if item not in kept),
    )
