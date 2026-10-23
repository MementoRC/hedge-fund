"""Tests for the JSON account snapshot store."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from hedge_fund.adapters.account.json_store import JsonSnapshotStore
from hedge_fund.domain.account import AccountSnapshot
from hedge_fund.domain.orders import Holding

EASTERN = timezone(timedelta(hours=-4))
SNAPSHOT = AccountSnapshot(
    as_of=date(2026, 10, 23),
    observed_at=datetime(2026, 10, 23, 8, 30, 0, 123456, tzinfo=EASTERN),
    equity=0.1 + 0.2,  # 0.30000000000000004: survives only an exact float round-trip
    holdings=(Holding(ticker="AAPL", shares=40), Holding(ticker="XOM", shares=-25)),
)
PAYLOAD = {
    "as_of": "2026-10-23",
    "observed_at": "2026-10-23T08:30:00.123456-04:00",
    "equity": 0.30000000000000004,
    "holdings": [["AAPL", 40], ["XOM", -25]],
}


def _sha12(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _files(root: Path) -> list[Path]:
    return sorted((root / "account").iterdir())


# --- save -------------------------------------------------------------------------------


def test_save_writes_the_payload_under_its_digest(tmp_path: Path) -> None:
    JsonSnapshotStore(tmp_path).save(SNAPSHOT)
    [path] = _files(tmp_path)
    assert re.fullmatch(r"2026-10-23_[0-9a-f]{12}\.json", path.name)
    assert path.name == f"2026-10-23_{_sha12(PAYLOAD)}.json"
    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["snapshot"] == PAYLOAD
    assert datetime.fromisoformat(body["created_at"]).utcoffset() == timedelta(0)


def test_saving_twice_leaves_one_file(tmp_path: Path) -> None:
    store = JsonSnapshotStore(tmp_path)
    store.save(SNAPSHOT)
    store.save(SNAPSHOT)
    assert len(_files(tmp_path)) == 1


def test_an_existing_file_is_never_overwritten(tmp_path: Path) -> None:
    store = JsonSnapshotStore(tmp_path)
    store.save(SNAPSHOT)
    [path] = _files(tmp_path)
    body = json.loads(path.read_text(encoding="utf-8"))
    body["created_at"] = "1999-01-01T00:00:00+00:00"  # outside the digest: still valid
    path.write_text(json.dumps(body), encoding="utf-8")
    store.save(SNAPSHOT)
    assert json.loads(path.read_text(encoding="utf-8"))["created_at"] == "1999-01-01T00:00:00+00:00"


def test_each_read_gets_its_own_file(tmp_path: Path) -> None:
    store = JsonSnapshotStore(tmp_path)
    later = dataclasses.replace(SNAPSHOT, observed_at=SNAPSHOT.observed_at + timedelta(seconds=1))
    store.save(SNAPSHOT)
    store.save(later)
    assert len(_files(tmp_path)) == 2


def test_save_rejects_a_non_finite_equity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Out of range float"):
        JsonSnapshotStore(tmp_path).save(dataclasses.replace(SNAPSHOT, equity=math.nan))
    assert not (tmp_path / "account").exists()


def test_package_exports() -> None:
    import hedge_fund.adapters.account as account

    assert account.__all__ == ["JsonSnapshotStore"]
    assert account.JsonSnapshotStore is JsonSnapshotStore
