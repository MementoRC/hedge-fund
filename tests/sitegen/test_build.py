"""sitegen.build: stats files, the --out safety rules, and the full build of a root."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hedge_fund.sitegen.build import MARKER, SiteError, check_out, read_stats
from hedge_fund.sitegen.pages import JunitCounts, Stats

JUNIT = (
    '<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" errors="0" '
    'failures="1" skipped="2" tests="12" time="1.0"/></testsuites>'
)
BARE_JUNIT = '<testsuite name="pytest" errors="1" failures="0" skipped="0" tests="3"/>'
ENTITY_JUNIT = (
    '<?xml version="1.0"?>\n<!DOCTYPE testsuites [<!ENTITY boom "x">]>\n'
    '<testsuites><testsuite errors="0" failures="0" skipped="0" tests="1"/></testsuites>'
)
DTD_JUNIT = (
    '<?xml version="1.0"?>\n<!DOCTYPE testsuites>\n'
    '<testsuites><testsuite errors="0" failures="0" skipped="0" tests="1"/></testsuites>'
)


def _stats_dir(tmp_path: Path, junit: str | None = None, coverage: object = None) -> Path:
    stats = tmp_path / "stats"
    stats.mkdir(parents=True, exist_ok=True)
    if junit is not None:
        (stats / "junit.xml").write_text(junit, encoding="utf-8")
    if coverage is not None:
        text = coverage if isinstance(coverage, str) else json.dumps(coverage)
        (stats / "coverage.json").write_text(text, encoding="utf-8")
    return stats


def _suite(tests: int, failures: int, errors: int, skipped: int) -> str:
    return (
        f'<testsuites><testsuite errors="{errors}" failures="{failures}" '
        f'skipped="{skipped}" tests="{tests}"/></testsuites>'
    )


def test_read_stats_reads_the_testsuite_under_testsuites_and_the_coverage(tmp_path: Path) -> None:
    stats = _stats_dir(tmp_path, JUNIT, {"totals": {"percent_covered": 99.996}})

    assert read_stats(stats) == Stats(JunitCounts(12, 1, 0, 2), 99.996)


def test_read_stats_accepts_a_testsuite_root(tmp_path: Path) -> None:
    assert read_stats(_stats_dir(tmp_path, BARE_JUNIT)).junit == JunitCounts(3, 0, 1, 0)


def test_missing_files_are_not_produced(tmp_path: Path) -> None:
    assert read_stats(_stats_dir(tmp_path)) == Stats(None, None)


@pytest.mark.parametrize(
    "junit",
    [
        "<testsuites",
        "<testsuites><other/></testsuites>",
        '<testsuites><testsuite tests="1"/></testsuites>',
        '<testsuites><testsuite errors="x" failures="0" skipped="0" tests="1"/></testsuites>',
    ],
)
def test_a_malformed_junit_is_not_produced(tmp_path: Path, junit: str) -> None:
    assert read_stats(_stats_dir(tmp_path, junit)).junit is None


@pytest.mark.parametrize("junit", [ENTITY_JUNIT, DTD_JUNIT])
def test_a_junit_with_a_dtd_or_an_entity_is_rejected(tmp_path: Path, junit: str) -> None:
    assert read_stats(_stats_dir(tmp_path, junit)).junit is None


@pytest.mark.parametrize(
    "junit",
    [
        _suite(-1, 0, 0, 0),
        _suite(5, -1, 0, 0),
        _suite(5, 0, -1, 0),
        _suite(5, 0, 0, -1),
        _suite(5, 3, 2, 1),
        _suite(0, 1, 0, 0),
    ],
)
def test_negative_or_inconsistent_junit_counts_are_not_produced(tmp_path: Path, junit: str) -> None:
    assert read_stats(_stats_dir(tmp_path, junit)).junit is None


def test_counts_that_add_up_exactly_are_produced(tmp_path: Path) -> None:
    junit = read_stats(_stats_dir(tmp_path, _suite(6, 3, 2, 1))).junit

    assert junit == JunitCounts(6, 3, 2, 1)


@pytest.mark.parametrize(
    "coverage",
    [
        "{",
        {"totals": {}},
        {"other": 1},
        ["x"],
        {"totals": {"percent_covered": "x"}},
        {"totals": {"percent_covered": None}},
        {"totals": {"percent_covered": True}},
        '{"totals": {"percent_covered": NaN}}',
        '{"totals": {"percent_covered": Infinity}}',
        '{"totals": {"percent_covered": -Infinity}}',
        '{"totals": {"percent_covered": 1e999}}',
    ],
)
def test_a_malformed_coverage_is_not_produced(tmp_path: Path, coverage: object) -> None:
    assert read_stats(_stats_dir(tmp_path, coverage=coverage)).coverage is None


def test_an_integer_coverage_is_a_float(tmp_path: Path) -> None:
    stats = _stats_dir(tmp_path, coverage={"totals": {"percent_covered": 100}})

    assert read_stats(stats).coverage == 100.0


def test_check_out_refuses_the_filesystem_root(demo_root: Path) -> None:
    with pytest.raises(SiteError, match="filesystem root"):
        check_out(Path("/"), demo_root)


def test_check_out_refuses_the_home_directory(
    demo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    with pytest.raises(SiteError, match="home directory"):
        check_out(home, demo_root)


def test_check_out_refuses_dot_and_a_dir_that_contains_the_cwd(
    demo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / "a" / "b"
    work.mkdir(parents=True)
    monkeypatch.chdir(work)

    with pytest.raises(SiteError, match="current directory"):
        check_out(Path("."), demo_root)
    with pytest.raises(SiteError, match="current directory"):
        check_out(tmp_path / "a", demo_root)


def test_check_out_refuses_the_root_and_a_dir_that_contains_it(demo_root: Path) -> None:
    with pytest.raises(SiteError, match="--root"):
        check_out(demo_root, demo_root)
    with pytest.raises(SiteError, match="--root"):
        check_out(demo_root.parent, demo_root)


def test_check_out_resolves_symlinks_and_dotdot_before_comparing(
    demo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    work = tmp_path / "work"
    home.mkdir()
    work.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(work)
    (tmp_path / "to_home").symlink_to(home, target_is_directory=True)
    (tmp_path / "to_root").symlink_to(demo_root, target_is_directory=True)
    (tmp_path / "to_slash").symlink_to(Path("/"), target_is_directory=True)

    with pytest.raises(SiteError, match="home directory"):
        check_out(tmp_path / "to_home", demo_root)
    with pytest.raises(SiteError, match="--root"):
        check_out(tmp_path / "to_root", demo_root)
    with pytest.raises(SiteError, match="filesystem root"):
        check_out(tmp_path / "to_slash", demo_root)
    with pytest.raises(SiteError, match="current directory"):
        check_out(work / "sub" / ".." / ".." / "work", demo_root)
    with pytest.raises(SiteError, match="home directory"):
        check_out(work / ".." / "home", demo_root)


def test_check_out_refuses_a_non_empty_dir_without_the_marker(
    demo_root: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("x", encoding="utf-8")

    with pytest.raises(SiteError, match=MARKER):
        check_out(out, demo_root)
    assert (out / "keep.txt").is_file()


def test_check_out_refuses_a_symlinked_marker(demo_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    real = tmp_path / "real_marker"
    real.write_text("", encoding="utf-8")
    (out / MARKER).symlink_to(real)
    (out / "keep.txt").write_text("x", encoding="utf-8")

    with pytest.raises(SiteError, match=MARKER):
        check_out(out, demo_root)


def test_check_out_refuses_a_marker_that_is_a_directory(demo_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    (out / MARKER).mkdir(parents=True)

    with pytest.raises(SiteError, match=MARKER):
        check_out(out, demo_root)


def test_check_out_reports_an_unreadable_dir_as_a_site_error(
    demo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    target = out.resolve()
    real_iterdir = Path.iterdir

    def deny(self: Path):  # noqa: ANN202
        if self == target:
            raise PermissionError("denied")
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", deny)

    with pytest.raises(SiteError, match="cannot read") as info:
        check_out(out, demo_root)
    assert str(out) in str(info.value)
    assert isinstance(info.value.__cause__, PermissionError)


def test_check_out_refuses_a_file(demo_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.write_text("x", encoding="utf-8")

    with pytest.raises(SiteError, match="not a directory"):
        check_out(out, demo_root)


def test_check_out_accepts_an_absent_an_empty_or_a_marked_dir(
    demo_root: Path, tmp_path: Path
) -> None:
    check_out(tmp_path / "absent", demo_root)
    (tmp_path / "empty").mkdir()
    check_out(tmp_path / "empty", demo_root)
    marked = tmp_path / "marked"
    marked.mkdir()
    (marked / MARKER).write_text("", encoding="utf-8")
    (marked / "old.md").write_text("x", encoding="utf-8")
    check_out(marked, demo_root)
