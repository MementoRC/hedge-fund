"""Empty states name the command that fills the page."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from hedge_fund.dashboard.states import (
    empty_message,
    missing_message,
)
from tests.dashboard.scripts import usable_probe


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
        ("refused", None, "usable"),  # a RootError alone keeps the workspace usable
    ],
)
def test_usable_workspace_explains_what_is_missing(
    case: str, info: str | None, caption: str
) -> None:
    at = AppTest.from_function(usable_probe, default_timeout=60, args=(case,)).run()

    assert len(at.exception) == 0
    # The probe reports its verdict as a caption ("usable" or "unusable"); the page's return
    # value cannot be read from outside the script, so the caption stands in for it.
    assert [c.value for c in at.caption] == [caption]
    if info is None:
        assert len(at.info) == 0
    else:
        assert info in at.info[0].value
        assert "pixi run -e dev demo" in at.info[0].value


def test_a_read_error_shows_an_error_and_no_data() -> None:
    at = AppTest.from_function(usable_probe, default_timeout=60, args=("read_error",)).run()

    assert len(at.exception) == 0
    assert [c.value for c in at.caption] == ["unusable"]
    assert len(at.info) == 0
    assert len(at.error) == 1
    assert "denied" in at.error[0].value
