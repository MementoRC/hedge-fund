"""``hedge-fund site build``: writes the generated pages; refusals exit 1 with one line."""

from __future__ import annotations

from pathlib import Path

import click
import pytest
from typer.testing import CliRunner, Result

from hedge_fund import cli
from hedge_fund.sitegen.build import MARKER
from hedge_fund.sitegen.pages import PAGE_FILES, STATS_PAGE
from tests.dashboard.builders import live_root

runner = CliRunner()


def _site(*args: str) -> Result:
    return runner.invoke(cli.app, ["site", "build", *args])


def test_site_build_writes_the_pages(demo_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    stats = tmp_path / "stats"
    stats.mkdir()

    result = _site("--root", str(demo_root), "--out", str(out), "--stats-dir", str(stats))

    assert result.exit_code == 0, result.output
    assert f"written to {out}" in click.unstyle(result.output)
    assert all((out / name).is_file() for name in PAGE_FILES)


def test_the_defaults_write_website_docs_generated_and_read_stats_from_the_cwd(
    demo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    coverage = '{"totals": {"percent_covered": 50.0}}'
    (tmp_path / "coverage.json").write_text(coverage, encoding="utf-8")

    result = _site("--root", str(demo_root))

    assert result.exit_code == 0, result.output
    out = tmp_path / "website" / "docs" / "generated"
    assert (out / MARKER).is_file()
    page = (out / STATS_PAGE).read_text(encoding="utf-8")
    assert "Coverage: 50.00%, below the 100% gate." in page


def test_a_live_root_is_refused_with_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = _site("--root", str(live_root(tmp_path)), "--out", str(tmp_path / "out"))

    assert result.exit_code == 1
    output = click.unstyle(result.output)
    assert output.startswith("error: ")
    assert output.count("\n") == 1
    assert not (tmp_path / "out").exists()


def test_an_unmarked_out_is_refused(
    demo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("x", encoding="utf-8")

    result = _site("--root", str(demo_root), "--out", str(out))

    assert result.exit_code == 1
    assert MARKER in click.unstyle(result.output)
    assert (out / "keep.txt").is_file()


def test_an_oserror_while_writing_is_one_line_not_a_traceback(
    demo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(cli, "build_site", boom)

    result = _site("--root", str(demo_root), "--out", str(tmp_path / "out"))

    assert result.exit_code != 0
    output = click.unstyle(result.output)
    assert "disk full" in output
    assert "Traceback" not in output


def test_root_is_required() -> None:
    result = _site()

    assert result.exit_code == 2
    assert "--root" in click.unstyle(result.output)
