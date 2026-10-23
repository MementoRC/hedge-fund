"""Tests for the JSON account snapshot store."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from hedge_fund.adapters.account.json_store import JsonSnapshotStore
from hedge_fund.domain.account import AccountSnapshot
from hedge_fund.domain.orders import Holding
from hedge_fund.ports.account import SnapshotStore, SnapshotStoreError

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


def _forge(root: Path, payload: Any, *, as_of: str = "2026-10-23") -> Path:
    """Write a file whose name matches ``payload``, so only the payload itself is wrong."""
    path = root / "account" / f"{as_of}_{_sha12(payload)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"created_at": "2026-10-23T12:00:00+00:00", "snapshot": payload}
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


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


# --- load -------------------------------------------------------------------------------


def test_satisfies_the_port_and_round_trips_exactly(tmp_path: Path) -> None:
    store: SnapshotStore = JsonSnapshotStore(tmp_path)
    store.save(SNAPSHOT)
    [loaded] = store.snapshots()
    assert loaded == SNAPSHOT
    assert loaded.equity == 0.30000000000000004
    assert loaded.observed_at.utcoffset() == timedelta(hours=-4)
    assert [holding.ticker for holding in loaded.holdings] == ["AAPL", "XOM"]


def test_snapshots_come_back_in_file_name_order(tmp_path: Path) -> None:
    store = JsonSnapshotStore(tmp_path)
    earlier = dataclasses.replace(
        SNAPSHOT,
        as_of=date(2026, 10, 22),
        observed_at=datetime(2026, 10, 22, 13, 0, tzinfo=UTC),
        equity=100.0,
        holdings=(),
    )
    store.save(SNAPSHOT)
    store.save(earlier)
    assert store.snapshots() == (earlier, SNAPSHOT)


def test_an_int_equity_loads_as_a_float(tmp_path: Path) -> None:
    _forge(tmp_path, {**PAYLOAD, "equity": 250})
    [loaded] = JsonSnapshotStore(tmp_path).snapshots()
    assert type(loaded.equity) is float
    assert loaded.equity == 250.0


def test_a_missing_directory_gives_no_snapshots(tmp_path: Path) -> None:
    assert JsonSnapshotStore(tmp_path / "nowhere").snapshots() == ()


def test_other_files_are_ignored(tmp_path: Path) -> None:
    store = JsonSnapshotStore(tmp_path)
    store.save(SNAPSHOT)
    (tmp_path / "account" / "notes.txt").write_text("not a snapshot", encoding="utf-8")
    assert store.snapshots() == (SNAPSHOT,)


def test_rejects_tampered_content(tmp_path: Path) -> None:
    store = JsonSnapshotStore(tmp_path)
    store.save(SNAPSHOT)
    [path] = _files(tmp_path)
    body = json.loads(path.read_text(encoding="utf-8"))
    body["snapshot"]["equity"] = 1_000_000.0
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(SnapshotStoreError, match=f"{re.escape(path.name)}: digest mismatch"):
        store.snapshots()


def test_rejects_a_name_whose_as_of_differs_from_the_payload(tmp_path: Path) -> None:
    path = _forge(tmp_path, PAYLOAD, as_of="2026-10-22")
    with pytest.raises(SnapshotStoreError, match=f"{re.escape(path.name)}: as_of mismatch"):
        JsonSnapshotStore(tmp_path).snapshots()


def test_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "account" / f"2026-10-23_{_sha12(PAYLOAD)}.json"
    path.parent.mkdir()
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SnapshotStoreError, match=f"{re.escape(path.name)}: malformed"):
        JsonSnapshotStore(tmp_path).snapshots()


def test_rejects_a_body_without_a_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "account" / f"2026-10-23_{_sha12(PAYLOAD)}.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"created_at": "x"}), encoding="utf-8")
    with pytest.raises(SnapshotStoreError, match=f"{re.escape(path.name)}: malformed"):
        JsonSnapshotStore(tmp_path).snapshots()


def test_rejects_an_unreadable_entry(tmp_path: Path) -> None:
    path = tmp_path / "account" / f"2026-10-23_{_sha12(PAYLOAD)}.json"
    path.mkdir(parents=True)  # a directory where a file should be
    with pytest.raises(SnapshotStoreError, match=f"{re.escape(path.name)}: malformed"):
        JsonSnapshotStore(tmp_path).snapshots()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("equity", True, id="bool-equity"),
        pytest.param("equity", "100.0", id="str-equity"),
        pytest.param("equity", None, id="null-equity"),
        pytest.param("observed_at", "2026-10-23T08:30:00", id="naive-observed_at"),
        pytest.param("observed_at", "yesterday", id="unparseable-observed_at"),
        pytest.param("observed_at", 1761222600, id="int-observed_at"),
        pytest.param("as_of", 20261023, id="int-as_of"),
        pytest.param("holdings", {}, id="holdings-object"),
        pytest.param("holdings", "", id="holdings-string"),
        pytest.param("holdings", [["AAPL", 1.5]], id="float-shares"),
        pytest.param("holdings", [["AAPL", True]], id="bool-shares"),
        pytest.param("holdings", [["AAPL", "40"]], id="str-shares"),
        pytest.param("holdings", [[40, "AAPL"]], id="int-ticker"),
        pytest.param("holdings", [["AAPL"]], id="short-pair"),
        pytest.param("holdings", [["AAPL", 40, 1]], id="long-pair"),
        pytest.param("holdings", [{"AAPL": 40}], id="pair-object"),
    ],
)
def test_rejects_a_mistyped_field(tmp_path: Path, field: str, value: Any) -> None:
    path = _forge(tmp_path, {**PAYLOAD, field: value})
    with pytest.raises(SnapshotStoreError, match=f"{re.escape(path.name)}: malformed"):
        JsonSnapshotStore(tmp_path).snapshots()


@pytest.mark.parametrize("field", ["as_of", "observed_at", "equity", "holdings"])
def test_rejects_a_missing_field(tmp_path: Path, field: str) -> None:
    payload = {key: value for key, value in PAYLOAD.items() if key != field}
    path = _forge(tmp_path, payload)
    with pytest.raises(SnapshotStoreError, match=f"{re.escape(path.name)}: malformed"):
        JsonSnapshotStore(tmp_path).snapshots()


def _raw(root: Path, text: str, name: str | None = None) -> Path:
    path = root / "account" / (name or f"2026-10-23_{_sha12(PAYLOAD)}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_rejects_an_equity_too_large_for_a_float(tmp_path: Path) -> None:
    path = _forge(tmp_path, {**PAYLOAD, "equity": 10**400})
    with pytest.raises(SnapshotStoreError, match=f"{re.escape(path.name)}: malformed"):
        JsonSnapshotStore(tmp_path).snapshots()


def test_rejects_a_body_nested_too_deeply_to_parse(tmp_path: Path) -> None:
    path = _raw(tmp_path, "[" * 100_000)
    with pytest.raises(SnapshotStoreError, match=f"{re.escape(path.name)}: malformed"):
        JsonSnapshotStore(tmp_path).snapshots()


def test_rejects_a_non_canonical_as_of(tmp_path: Path) -> None:
    path = _forge(tmp_path, {**PAYLOAD, "as_of": "20261023"}, as_of="20261023")
    with pytest.raises(SnapshotStoreError, match=f"{re.escape(path.name)}: malformed"):
        JsonSnapshotStore(tmp_path).snapshots()


@pytest.mark.parametrize("observed_at", ["2026-10-23T13:00:00+0000", "2026-10-23T13:00:00Z"])
def test_rejects_a_non_canonical_observed_at(tmp_path: Path, observed_at: str) -> None:
    path = _forge(tmp_path, {**PAYLOAD, "observed_at": observed_at})
    with pytest.raises(SnapshotStoreError, match=f"{re.escape(path.name)}: malformed"):
        JsonSnapshotStore(tmp_path).snapshots()


@pytest.mark.parametrize("payload", [{**PAYLOAD, "extra": 1}], ids=["extra-key"])
def test_rejects_unexpected_keys(tmp_path: Path, payload: dict[str, Any]) -> None:
    path = _forge(tmp_path, payload)
    with pytest.raises(SnapshotStoreError, match=f"{re.escape(path.name)}: malformed"):
        JsonSnapshotStore(tmp_path).snapshots()


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_rejects_a_non_finite_equity_token(tmp_path: Path, token: str) -> None:
    text = json.dumps({"snapshot": {**PAYLOAD, "equity": 0}}).replace(
        '"equity": 0', f'"equity": {token}'
    )
    path = _raw(tmp_path, text)
    with pytest.raises(SnapshotStoreError, match=f"{re.escape(path.name)}: malformed"):
        JsonSnapshotStore(tmp_path).snapshots()


@pytest.mark.parametrize("text", ["[]", "null"])
def test_rejects_a_body_that_is_not_an_object(tmp_path: Path, text: str) -> None:
    path = _raw(tmp_path, text)
    with pytest.raises(SnapshotStoreError, match=f"{re.escape(path.name)}: malformed"):
        JsonSnapshotStore(tmp_path).snapshots()


@pytest.mark.parametrize("name", ["latest.json", "2026-10-23.json", "2026-10-23_ABCDEF123456.json"])
def test_rejects_a_json_file_whose_name_is_not_date_and_digest(tmp_path: Path, name: str) -> None:
    body = {"created_at": "x", "snapshot": PAYLOAD}
    path = _raw(tmp_path, json.dumps(body), name)
    with pytest.raises(SnapshotStoreError, match=re.escape(path.name)):
        JsonSnapshotStore(tmp_path).snapshots()


def test_one_bad_file_fails_the_whole_read(tmp_path: Path) -> None:
    store = JsonSnapshotStore(tmp_path)
    store.save(SNAPSHOT)
    _forge(tmp_path, {**PAYLOAD, "equity": True}, as_of="2026-10-23")
    with pytest.raises(SnapshotStoreError):
        store.snapshots()
