"""Tests for the write-once JSON fill store."""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from hedge_fund.adapters.ledger.fills import JsonFillStore
from hedge_fund.domain.ledger import Fill, FillStatus
from hedge_fund.domain.orders import OrderAction
from hedge_fund.ports.ledger import FillStore

OPEN = datetime(2026, 10, 1, 13, 30, tzinfo=UTC)
RECORDED = datetime(2026, 10, 1, 14, 0, tzinfo=UTC)
FILL = Fill(
    client_order_id="hf-2026-09-30-AAA-buy-10",
    broker_order_id="b-1",
    cycle_as_of=date(2026, 9, 30),
    ticker="AAA",
    action=OrderAction.BUY,
    requested_shares=10,
    filled_shares=10,
    avg_price=101.25,
    filled_at=OPEN,
    status=FillStatus.FILLED,
    recorded_at=RECORDED,
)
NEVER = dataclasses.replace(
    FILL,
    client_order_id="hf-2026-09-30-BRK-B-sell_short-5",
    broker_order_id=None,
    ticker="BRK-B",
    action=OrderAction.SELL_SHORT,
    requested_shares=5,
    filled_shares=0,
    avg_price=None,
    filled_at=None,
    status=FillStatus.NOT_SUBMITTED,
)
EXPECTED = {
    "client_order_id": "hf-2026-09-30-AAA-buy-10",
    "broker_order_id": "b-1",
    "cycle_as_of": "2026-09-30",
    "ticker": "AAA",
    "action": "buy",
    "requested_shares": 10,
    "filled_shares": 10,
    "avg_price": 101.25,
    "filled_at": "2026-10-01T13:30:00+00:00",
    "status": "filled",
    "recorded_at": "2026-10-01T14:00:00+00:00",
}


def _dir(root: Path) -> Path:
    return root / "ledger" / "fills"


def _path(root: Path, fill: Fill = FILL) -> Path:
    return _dir(root) / f"{fill.client_order_id}.json"


def _tamper(root: Path, **changes: Any) -> None:
    body = json.loads(_path(root).read_text(encoding="utf-8"))
    body.update(changes)
    _path(root).write_text(json.dumps(body), encoding="utf-8")


def test_satisfies_the_port_and_round_trips(tmp_path: Path) -> None:
    store: FillStore = JsonFillStore(tmp_path)
    store.save(NEVER)
    store.save(FILL)
    assert store.load_all() == (FILL, NEVER)  # by file name
    assert sorted(path.name for path in _dir(tmp_path).iterdir()) == [
        "hf-2026-09-30-AAA-buy-10.json",
        "hf-2026-09-30-BRK-B-sell_short-5.json",
    ]


def test_the_file_is_the_fill_in_field_order(tmp_path: Path) -> None:
    JsonFillStore(tmp_path).save(FILL)
    text = _path(tmp_path).read_text(encoding="utf-8")
    assert json.loads(text) == EXPECTED
    assert text == json.dumps(EXPECTED, indent=2)


def test_a_recorded_fill_is_never_overwritten(tmp_path: Path) -> None:
    store = JsonFillStore(tmp_path)
    store.save(FILL)
    first = _path(tmp_path).read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match=FILL.client_order_id):
        store.save(dataclasses.replace(FILL, filled_shares=3))
    assert _path(tmp_path).read_text(encoding="utf-8") == first


def test_a_failed_write_leaves_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_text = Path.write_text

    def partial_then_fail(self: Path, data: str, *args: Any, **kwargs: Any) -> int:
        write_text(self, data[:10], *args, **kwargs)
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", partial_then_fail)
    with pytest.raises(OSError, match="disk full"):
        JsonFillStore(tmp_path).save(FILL)
    assert list(_dir(tmp_path).iterdir()) == []


def test_a_leftover_temp_file_is_ignored(tmp_path: Path) -> None:
    store = JsonFillStore(tmp_path)
    store.save(FILL)
    (_dir(tmp_path) / f".{FILL.client_order_id}.json.k3j2.tmp").write_text("{", encoding="utf-8")
    assert store.load_all() == (FILL,)


def test_nothing_recorded_loads_nothing(tmp_path: Path) -> None:
    assert JsonFillStore(tmp_path).load_all() == ()


def test_a_file_named_after_another_id_raises(tmp_path: Path) -> None:
    JsonFillStore(tmp_path).save(FILL)
    _path(tmp_path).rename(_dir(tmp_path) / "hf-other.json")
    with pytest.raises(ValueError, match="hf-other.json: holds fill 'hf-2026-09-30-AAA-buy-10'"):
        JsonFillStore(tmp_path).load_all()


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({"status": "bogus"}, id="unknown-status"),
        pytest.param({"filled_shares": "10"}, id="string-shares"),
        pytest.param({"filled_shares": True}, id="bool-shares"),
        pytest.param({"filled_at": "2026-10-01T13:30:00"}, id="naive-filled-at"),
        pytest.param({"avg_price": "101.25"}, id="string-price"),
        pytest.param({"extra": 1}, id="extra-key"),
        pytest.param({"ticker": ""}, id="empty-ticker"),
    ],
)
def test_tampered_content_raises(tmp_path: Path, changes: dict[str, Any]) -> None:
    JsonFillStore(tmp_path).save(FILL)
    _tamper(tmp_path, **changes)
    with pytest.raises(ValueError, match=f"{FILL.client_order_id}.json: malformed fill record"):
        JsonFillStore(tmp_path).load_all()


@pytest.mark.parametrize(
    "text", [pytest.param("not json", id="not-json"), pytest.param("[]", id="not-an-object")]
)
def test_a_body_that_is_not_a_fill_object_raises(tmp_path: Path, text: str) -> None:
    JsonFillStore(tmp_path).save(FILL)
    _path(tmp_path).write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="malformed fill record"):
        JsonFillStore(tmp_path).load_all()


@pytest.mark.parametrize("bad", ["", ".hidden", "a/b", "..", "../escape"])
def test_save_refuses_ids_outside_the_pattern(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError, match="is not a valid client_order_id"):
        JsonFillStore(tmp_path).save(dataclasses.replace(FILL, client_order_id=bad))
    assert not _dir(tmp_path).exists()


@pytest.mark.parametrize("name", [".json", ".hidden.json", "..json"])
def test_load_all_refuses_file_names_outside_the_pattern(tmp_path: Path, name: str) -> None:
    _dir(tmp_path).mkdir(parents=True)
    (_dir(tmp_path) / name).write_text(json.dumps(EXPECTED), encoding="utf-8")
    with pytest.raises(ValueError, match="is not a valid client_order_id"):
        JsonFillStore(tmp_path).load_all()
