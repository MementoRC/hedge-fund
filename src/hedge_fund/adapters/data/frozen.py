"""Content-addressed frozen Price- and FundamentalsSnapshot files for reproducible runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from hedge_fund.adapters.data._atomic import atomic_write
from hedge_fund.domain.fundamentals import FundamentalsSnapshot
from hedge_fund.domain.snapshot import PriceSnapshot

FUNDAMENTALS_DIR = "fundamentals"


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
    """Read a saved pair, verifying its content digest against the file name.

    Constructing PriceSnapshot revalidates its contract.
    """
    parquet = Path(path).with_suffix(".parquet")
    meta = json.loads(parquet.with_suffix(".json").read_text(encoding="utf-8"))
    closes = pd.read_parquet(parquet)[meta["columns"]]
    as_of = date.fromisoformat(meta["as_of"])
    actual = _digest(as_of, meta["columns"], meta["sectors"], closes)
    expected = parquet.stem.rsplit("_", 1)[-1]
    if actual != expected:
        raise ValueError(
            f"{parquet.name}: digest mismatch, name has {expected}, content is {actual}"
        )
    return PriceSnapshot(as_of=as_of, closes=closes, sectors=meta["sectors"])


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
    digest.update(_canonical_floats(closes).tobytes())
    return digest.hexdigest()[:12]


def _canonical_floats(frame: pd.DataFrame) -> np.ndarray:
    """C-ordered float64 values with canonical NaN and +0.0, so equal data hashes identically."""
    values = np.array(frame.to_numpy(dtype="float64"), dtype="float64", order="C")
    return np.asarray(np.where(np.isnan(values), np.nan, values) + 0.0, dtype="float64")


def save_fundamentals(snapshot: FundamentalsSnapshot, root: Path) -> Path:
    """Write fundamentals/<as_of>_<sha12>.parquet and .json under root; return the parquet path.

    One parquet holds current and prior metrics and the periods dates under a two-level
    column index (current/prior x metric, periods x date column). As with price snapshots,
    the name derives from the content and an existing pair is returned as-is.
    """
    metrics = [str(column) for column in snapshot.current.columns]
    periods = [str(column) for column in snapshot.periods.columns]
    combined = pd.concat([snapshot.current, snapshot.prior, snapshot.periods], axis=1).set_axis(
        _fundamentals_columns(metrics, periods), axis=1
    )
    digest = _fundamentals_digest(snapshot.as_of, combined)
    parquet = Path(root) / FUNDAMENTALS_DIR / f"{snapshot.as_of.isoformat()}_{digest}.parquet"
    meta = parquet.with_suffix(".json")
    if parquet.is_file() and meta.is_file():
        return parquet
    atomic_write(parquet, combined.to_parquet)
    body = json.dumps(
        {
            "as_of": snapshot.as_of.isoformat(),
            "metrics": metrics,
            "periods": periods,
            "created_at": datetime.now(UTC).isoformat(),
        },
        indent=2,
    )
    atomic_write(meta, lambda path: path.write_text(body, encoding="utf-8"))
    return parquet


def load_fundamentals(path: Path) -> FundamentalsSnapshot:
    """Read a saved pair, verifying its content digest against the file name.

    The column index is rebuilt from the sidecar; constructing FundamentalsSnapshot
    revalidates its contract. Periods are cast back to datetime64[ns] so dtypes match the
    saved snapshot whatever time unit pyarrow returns.
    """
    parquet = Path(path).with_suffix(".parquet")
    meta = json.loads(parquet.with_suffix(".json").read_text(encoding="utf-8"))
    columns = _fundamentals_columns(meta["metrics"], meta["periods"])
    combined = pd.read_parquet(parquet).set_axis(columns, axis=1)
    as_of = date.fromisoformat(meta["as_of"])
    name_as_of, expected = parquet.stem.rsplit("_", 1)
    if name_as_of != as_of.isoformat():
        raise ValueError(
            f"{parquet.name}: as_of mismatch, name has {name_as_of}, sidecar has {as_of}"
        )
    actual = _fundamentals_digest(as_of, combined)
    if actual != expected:
        raise ValueError(
            f"{parquet.name}: digest mismatch, name has {expected}, content is {actual}"
        )
    return FundamentalsSnapshot(
        as_of=as_of,
        current=combined["current"],
        prior=combined["prior"],
        periods=combined["periods"].astype("datetime64[ns]"),
    )


def _fundamentals_columns(metrics: Sequence[str], periods: Sequence[str]) -> pd.MultiIndex:
    tuples = [(block, name) for block in ("current", "prior") for name in metrics]
    tuples += [("periods", name) for name in periods]
    return pd.MultiIndex.from_tuples(tuples)


def _fundamentals_digest(as_of: date, combined: pd.DataFrame) -> str:
    """First 12 hex chars of SHA-256 over as_of, the sorted index, columns and all values."""
    header = json.dumps(
        {
            "as_of": as_of.isoformat(),
            "index": [str(ticker) for ticker in combined.index],
            "columns": [list(column) for column in combined.columns],
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(header.encode("utf-8"))
    for block in ("current", "prior"):
        digest.update(_canonical_floats(combined[block]).tobytes())
    dates = combined["periods"].astype("datetime64[ns]").to_numpy(dtype="datetime64[ns]")
    digest.update(np.ascontiguousarray(dates).view("int64").tobytes())
    return digest.hexdigest()[:12]
