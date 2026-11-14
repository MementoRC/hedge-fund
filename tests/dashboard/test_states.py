"""Empty states name the command that fills the page."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from hedge_fund.dashboard.states import (
    DEMO_COMMAND,
    empty_message,
    missing_message,
)
from tests.dashboard.scripts import usable_probe


def test_the_demo_command_is_the_pixi_task() -> None:
    assert DEMO_COMMAND == "pixi run -e dev demo"


def test_the_missing_message_names_the_root_and_the_demo_command() -> None:
    message = missing_message(Path("data/demo"))

    assert "`data/demo`" in message
    assert "`pixi run -e dev demo`" in message


def test_the_empty_message_names_every_filling_command() -> None:
    message = empty_message(Path("some/root"))

    assert "`some/root`" in message
    assert "`pixi run -e dev demo`" in message
    assert "`hedge-fund backtest run`" in message
    assert "`hedge-fund cycle`" in message
    assert "unreadable" in message


@pytest.mark.parametrize(
    ("case", "info", "caption"),
    [
        ("missing", "No store at", "unusable"),
        ("empty", "no account snapshots", "unusable"),
        ("usable", None, "usable"),
    ],
)
def test_usable_workspace_explains_what_is_missing(
    case: str, info: str | None, caption: str
) -> None:
    at = AppTest.from_function(usable_probe, default_timeout=60, args=(case,)).run()

    assert len(at.exception) == 0
    assert [c.value for c in at.caption] == [caption]
    if info is None:
        assert len(at.info) == 0
    else:
        assert info in at.info[0].value
        assert "pixi run -e dev demo" in at.info[0].value
