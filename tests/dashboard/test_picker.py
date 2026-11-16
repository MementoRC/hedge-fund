"""The row picker shared by the Cycles and Backtests pages."""

from __future__ import annotations

from typing import Any

import pytest

from hedge_fund.dashboard import picker
from hedge_fund.dashboard.picker import pick_index


def _stub(monkeypatch: pytest.MonkeyPatch, picked: int | None) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def selectbox(label: str, options: list[int], **kwargs: Any) -> int | None:
        seen.update(label=label, options=options, **kwargs)
        return picked

    monkeypatch.setattr(picker.st, "selectbox", selectbox)
    return seen


def test_the_pick_is_returned_and_the_widget_gets_the_order_the_labels_and_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _stub(monkeypatch, 0)

    index = pick_index("Run", [2, 1, 0], lambda i: f"row {i}", key="run:r")

    assert index == 0
    assert (seen["label"], seen["options"], seen["key"]) == ("Run", [2, 1, 0], "run:r")
    assert seen["format_func"](1) == "row 1"


@pytest.mark.parametrize("picked", [3, -1, None])
def test_an_out_of_range_or_missing_pick_falls_back(
    monkeypatch: pytest.MonkeyPatch, picked: int | None
) -> None:
    _stub(monkeypatch, picked)

    assert pick_index("Run", [2, 1, 0], str, key="run:r") == 2
