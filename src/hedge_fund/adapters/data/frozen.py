"""Content-addressed frozen PriceSnapshot files for reproducible evaluation runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from hedge_fund.adapters.data._atomic import atomic_write
from hedge_fund.domain.snapshot import PriceSnapshot


def save_snapshot(snapshot: PriceSnapshot, root: Path) -> Path:
    """Write <as_of>_<sha12>.parquet and .json under root; return the parquet path.

    The name is derived from the content (created_at excluded), so an existing pair is
    returned as-is and never overwritten.
    """
    columns = [str(column) for column in snapshot.closes.columns]
    sectors = {column: snapshot.sectors[column] for column in columns}
    digest = _digest(snapshot.as_of, columns, sectors, snapshot.closes)
    parquet = Path(root) / f"{snapshot.as_of.isoformat()}_{digest}.parquet"
    meta = parquet.with_suffix(".json")
    if parquet.is_file() and meta.is_file():
        return parquet
    atomic_write(parquet, snapshot.closes.to_parquet)
    body = json.dumps(
        {
            "as_of": snapshot.as_of.isoformat(),
            "columns": columns,
            "sectors": sectors,
            "created_at": datetime.now(UTC).isoformat(),
        },
        indent=2,
    )
    atomic_write(meta, lambda path: path.write_text(body, encoding="utf-8"))
    return parquet


def load_snapshot(path: Path) -> PriceSnapshot:
    """Read a saved pair; constructing PriceSnapshot revalidates its contract."""
    parquet = Path(path).with_suffix(".parquet")
    meta = json.loads(parquet.with_suffix(".json").read_text(encoding="utf-8"))
    closes = pd.read_parquet(parquet)[meta["columns"]]
    return PriceSnapshot(
        as_of=date.fromisoformat(meta["as_of"]),
        closes=closes,
        sectors=meta["sectors"],
    )


def _digest(
    as_of: date, columns: Sequence[str], sectors: Mapping[str, str], closes: pd.DataFrame
) -> str:
    """First 12 hex chars of SHA-256 over as_of, columns, sectors, index and values."""
    header = json.dumps(
        {"as_of": as_of.isoformat(), "columns": list(columns), "sectors": dict(sectors)},
        sort_keys=True,
    )
    digest = hashlib.sha256(header.encode("utf-8"))
    digest.update(pd.DatetimeIndex(closes.index).as_unit("ns").asi8.tobytes())
    digest.update(np.ascontiguousarray(closes.to_numpy(dtype="float64")).tobytes())
    return digest.hexdigest()[:12]
