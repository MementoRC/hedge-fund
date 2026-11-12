"""The DEMO.json marker: written last, read strictly."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hedge_fund.adapters.demo_marker import (
    DEMO_MARKER,
    demo_store_dir,
    read_demo_marker,
    write_demo_marker,
)
from hedge_fund.domain.demo import DemoMarker

MARKER = DemoMarker(seed=3, store_digest="a" * 64, run_id="2024-01-10_2024-03-08_abc123def456")


def test_the_marker_round_trips_with_the_documented_shape(tmp_path: Path) -> None:
    path = write_demo_marker(tmp_path, MARKER)

    assert path == tmp_path / DEMO_MARKER
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "kind": "demo",
        "format": 1,
        "seed": 3,
        "store_digest": "a" * 64,
        "run_id": "2024-01-10_2024-03-08_abc123def456",
    }
    assert read_demo_marker(tmp_path) == MARKER


def test_a_root_without_a_marker_reads_as_none(tmp_path: Path) -> None:
    assert read_demo_marker(tmp_path) is None


def test_the_marker_is_never_overwritten(tmp_path: Path) -> None:
    write_demo_marker(tmp_path, MARKER)

    with pytest.raises(FileExistsError):
        write_demo_marker(tmp_path, MARKER)


def test_the_store_sits_beside_the_root() -> None:
    assert demo_store_dir(Path("data/demo")) == Path("data/demo.store")


GOOD = {"kind": "demo", "format": 1, "seed": 1, "store_digest": "d", "run_id": "r"}


@pytest.mark.parametrize(
    "body",
    [
        "{",
        "[]",
        json.dumps({**GOOD, "extra": 1}),
        json.dumps({k: v for k, v in GOOD.items() if k != "seed"}),
        json.dumps({**GOOD, "kind": "backtest"}),
        json.dumps({**GOOD, "format": 2}),
        json.dumps({**GOOD, "seed": True}),
        json.dumps({**GOOD, "seed": "1"}),
        json.dumps({**GOOD, "store_digest": 5}),
        json.dumps({**GOOD, "run_id": None}),
    ],
)
def test_a_marker_that_is_not_exactly_a_demo_marker_is_refused(tmp_path: Path, body: str) -> None:
    (tmp_path / DEMO_MARKER).write_text(body, encoding="utf-8")

    with pytest.raises(ValueError, match="not a demo marker"):
        read_demo_marker(tmp_path)
