"""The frozen backtest store: content-addressed files and a SHA-256 manifest.

Layout under the store root: one ``<kind>_<sha12>.<parquet|json>`` file per kind in KINDS
and ``manifest.json``, written last with atomic_create, so an interrupted write leaves no
manifest and the store does not load. Each name carries the first 12 hex characters of the
SHA-256 of the file's bytes and the manifest the full digest; loading re-hashes every file.
Unlike frozen.py, which hashes decoded frames so a name exists before the write, the digest
here is over the exact bytes written.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import pandas as pd

from hedge_fund.adapters.data._atomic import atomic_create
from hedge_fund.domain.backtest import Dividend, FlaggedSpan, FrozenData, MemberSpan
from hedge_fund.domain.ledger import Split
from hedge_fund.ports.fundamentals import coerce_facts

STORE_FORMAT = 1
MANIFEST = "manifest.json"
KINDS: tuple[str, ...] = (
    "opens",
    "closes",
    "splits",
    "dividends",
    "facts",
    "membership",
    "flagged",
    "sectors",
    "ciks",
)
_MANIFEST_KEYS = (
    "format",
    "created_at",
    "coverage",
    "files",
    "unpriced",
    "no_sector",
    "no_cik",
    "sic_sectors",
)


class FrozenStoreError(ValueError):
    """A frozen store root that is not fresh, or a store that is incomplete or tampered."""


def check_fresh_root(root: Path) -> None:
    """Raise FrozenStoreError unless ``root`` is absent or an empty directory."""
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FrozenStoreError(f"{root} is not empty; a frozen store needs a fresh root")


def write_store(root: Path, data: FrozenData, *, created_at: datetime) -> str:
    """Write ``data`` under a fresh ``root``; return the manifest's SHA-256 hex digest.

    Raises FrozenStoreError, writing nothing, when ``root`` is not fresh.
    """
    check_fresh_root(root)
    files = {kind: _put(root, kind, suffix, body) for kind, (suffix, body) in _encode(data).items()}
    manifest = {
        "format": STORE_FORMAT,
        "created_at": created_at.isoformat(),
        "coverage": {"start": data.start.isoformat(), "end": data.end.isoformat()},
        "files": files,
        "unpriced": list(data.unpriced),
        "no_sector": list(data.no_sector),
        "no_cik": list(data.no_cik),
        "sic_sectors": list(data.sic_sectors),
    }
    body = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    atomic_create(root / MANIFEST, lambda path: path.write_bytes(body))
    return hashlib.sha256(body).hexdigest()


def load_store(root: Path) -> FrozenData:
    """Read the store under ``root``, verifying every file against the manifest.

    Raises FrozenStoreError on a missing or unreadable manifest, an unknown format, a
    missing key or file kind, a file name that is not a plain name of its kind, a missing
    file or a SHA-256 mismatch. Constructing FrozenData revalidates the price frames.
    """
    manifest = _manifest(root)
    files = manifest["files"]
    if sorted(files) != sorted(KINDS):
        raise FrozenStoreError(f"{root}: manifest files {sorted(files)} are not {sorted(KINDS)}")
    bodies = {kind: _verified(root, kind, files[kind]) for kind in KINDS}
    coverage = manifest["coverage"]
    return FrozenData(
        start=date.fromisoformat(coverage["start"]),
        end=date.fromisoformat(coverage["end"]),
        opens=_prices(bodies["opens"]),
        closes=_prices(bodies["closes"]),
        splits=_splits(_parquet(bodies["splits"])),
        dividends=_dividends(_parquet(bodies["dividends"])),
        facts=coerce_facts(_parquet(bodies["facts"])),
        sectors=json.loads(bodies["sectors"]),
        ciks=json.loads(bodies["ciks"]),
        membership=_membership(_parquet(bodies["membership"])),
        unpriced=tuple(manifest["unpriced"]),
        no_sector=tuple(manifest["no_sector"]),
        no_cik=tuple(manifest["no_cik"]),
        sic_sectors=tuple(manifest["sic_sectors"]),
        flagged=_flagged(_parquet(bodies["flagged"])),
    )


def _encode(data: FrozenData) -> dict[str, tuple[str, bytes]]:
    """Each kind's file suffix and bytes."""
    splits = pd.DataFrame(
        {
            "ticker": pd.Series([s.ticker for s in data.splits], dtype="str"),
            "ex_date": _dates([s.ex_date for s in data.splits]),
            "numerator": pd.Series([s.ratio.numerator for s in data.splits], dtype="int64"),
            "denominator": pd.Series([s.ratio.denominator for s in data.splits], dtype="int64"),
        }
    )
    dividends = pd.DataFrame(
        {
            "ticker": pd.Series([d.ticker for d in data.dividends], dtype="str"),
            "ex_date": _dates([d.ex_date for d in data.dividends]),
            "amount": pd.Series([d.amount for d in data.dividends], dtype="float64"),
        }
    )
    membership = pd.DataFrame(
        {
            "ticker": pd.Series([m.ticker for m in data.membership], dtype="str"),
            "start": _dates([m.start for m in data.membership]),
            "end": _dates([m.end for m in data.membership]),
        }
    )
    flagged = pd.DataFrame(
        {
            "ticker": pd.Series([f.span.ticker for f in data.flagged], dtype="str"),
            "start": _dates([f.span.start for f in data.flagged]),
            "end": _dates([f.span.end for f in data.flagged]),
            "reason": pd.Series([f.reason for f in data.flagged], dtype="str"),
        }
    )
    return {
        "opens": ("parquet", data.opens.to_parquet()),
        "closes": ("parquet", data.closes.to_parquet()),
        "splits": ("parquet", splits.to_parquet()),
        "dividends": ("parquet", dividends.to_parquet()),
        "facts": ("parquet", data.facts.to_parquet()),
        "membership": ("parquet", membership.to_parquet()),
        "flagged": ("parquet", flagged.to_parquet()),
        "sectors": ("json", _json(data.sectors)),
        "ciks": ("json", _json(data.ciks)),
    }


def _dates(values: Sequence[date | None]) -> pd.Series:
    return pd.to_datetime(pd.Series(list(values), dtype="object")).astype("datetime64[ns]")


def _json(mapping: Mapping[str, str]) -> bytes:
    return json.dumps(dict(sorted(mapping.items())), indent=2).encode("utf-8")


def _put(root: Path, kind: str, suffix: str, body: bytes) -> dict[str, str]:
    digest = hashlib.sha256(body).hexdigest()
    name = f"{kind}_{digest[:12]}.{suffix}"
    atomic_create(root / name, lambda path: path.write_bytes(body))
    return {"name": name, "sha256": digest}


def _manifest(root: Path) -> dict[str, Any]:
    path = root / MANIFEST
    if not path.is_file():
        raise FrozenStoreError(f"{root}: no {MANIFEST}; not a complete frozen store")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FrozenStoreError(f"{path}: unreadable manifest: {exc}") from exc
    if not isinstance(manifest, dict) or any(key not in manifest for key in _MANIFEST_KEYS):
        raise FrozenStoreError(f"{path}: manifest lacks one of {list(_MANIFEST_KEYS)}")
    if manifest["format"] != STORE_FORMAT:
        raise FrozenStoreError(f"{path}: format {manifest['format']!r}, expected {STORE_FORMAT}")
    return manifest


def _verified(root: Path, kind: str, entry: Mapping[str, str]) -> bytes:
    name, expected = entry["name"], entry["sha256"]
    if Path(name).name != name or not name.startswith(f"{kind}_"):
        raise FrozenStoreError(f"{root}: bad file name {name!r} for {kind}")
    path = root / name
    if not path.is_file():
        raise FrozenStoreError(f"{root}: {name} is missing")
    body = path.read_bytes()
    actual = hashlib.sha256(body).hexdigest()
    if actual != expected or not Path(name).stem.endswith(f"_{actual[:12]}"):
        raise FrozenStoreError(
            f"{path}: digest mismatch, manifest has {expected}, content is {actual}"
        )
    return body


def _parquet(body: bytes) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(body))


def _prices(body: bytes) -> pd.DataFrame:
    frame = _parquet(body)
    index = pd.DatetimeIndex(frame.index).as_unit("ns")
    return frame.set_axis(index, axis=0).astype("float64").rename_axis(index=None, columns=None)


def _splits(frame: pd.DataFrame) -> tuple[Split, ...]:
    return tuple(
        Split(
            str(row.ticker),
            row.ex_date.date(),
            Fraction(int(row.numerator), int(row.denominator)),
        )
        for row in frame.itertuples(index=False)
    )


def _dividends(frame: pd.DataFrame) -> tuple[Dividend, ...]:
    return tuple(
        Dividend(str(row.ticker), row.ex_date.date(), float(row.amount))
        for row in frame.itertuples(index=False)
    )


def _span(row: Any) -> MemberSpan:
    return MemberSpan(
        str(row.ticker), row.start.date(), None if pd.isna(row.end) else row.end.date()
    )


def _membership(frame: pd.DataFrame) -> tuple[MemberSpan, ...]:
    return tuple(_span(row) for row in frame.itertuples(index=False))


def _flagged(frame: pd.DataFrame) -> tuple[FlaggedSpan, ...]:
    return tuple(FlaggedSpan(_span(row), str(row.reason)) for row in frame.itertuples(index=False))
