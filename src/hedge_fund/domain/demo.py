"""Plain data for the demo world's marker file."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DemoMarker:
    """What ``<root>/DEMO.json`` records: the price seed, the SHA-256 of the store's
    ``manifest.json`` and the id of the one backtest run in the root.
    """

    seed: int
    store_digest: str
    run_id: str
