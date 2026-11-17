"""The demo build is offline: it never opens a network connection (site S2, Decisions 3).

The Pages workflow builds the demo world in CI, so a network call there would make the site
depend on a vendor. Every socket entry point raises instead of connecting.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hedge_fund import cli


class NetworkUsedError(BaseException):
    """A socket call during the demo build.

    A BaseException, not an Exception: the broad ``except Exception`` blocks in ``cli.py``
    (demo store write) and ``viz/load.py`` must not be able to swallow it.
    """


def _refuse(*args: object, **kwargs: object) -> None:
    raise NetworkUsedError(f"network call during the demo build: {args!r}")


def test_the_demo_build_makes_no_network_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)
    monkeypatch.setattr(socket, "getaddrinfo", _refuse)

    result = CliRunner().invoke(cli.app, ["demo", "build", "--root", str(tmp_path / "demo")])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "demo" / "DEMO.json").is_file()
