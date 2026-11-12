"""Import rules between the layers, checked on the source tree (design spec, Layering)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "hedge_fund"
PURE = ("domain", "engine", "demo", "viz/views.py", "viz/charts.py", "viz/theme.py")
PURE_BANNED = (
    "hedge_fund.adapters",
    "hedge_fund.services",
    "hedge_fund.cli",
    "streamlit",
    "mkdocs",
)
IO_MODULES = frozenset(
    {"subprocess", "socket", "requests", "urllib", "http", "shutil", "tempfile", "sqlite3"}
)
IO_CALLS = frozenset(
    {"open", "read_text", "write_text", "read_bytes", "write_bytes", "mkdir", "unlink", "rmtree"}
)
ADAPTER_IMPORTERS = frozenset({"cli.py", "viz/load.py"})
BROKER_IMPORTERS = frozenset({"cli.py"})


def _under(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(f"{prefix}.")


def _resolve(path: Path, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    package = ["hedge_fund", *path.relative_to(SRC).parent.parts]
    base = package[: len(package) - (node.level - 1)]
    return ".".join([*base, *([node.module] if node.module else [])])


def _imports(path: Path) -> set[str]:
    """Every module a file imports, plus ``module.name`` for each ``from module import name``."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = _resolve(path, node)
            found.add(module)
            found.update(f"{module}.{alias.name}" for alias in node.names)
    return found


def _io_imports(path: Path) -> set[str]:
    """Imported names under an I/O module, by prefix (``urllib.request``, ``http.client.X``)."""
    return {name for name in _imports(path) if any(_under(name, module) for module in IO_MODULES)}


def _calls(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call):
            func = node.func
            names.add(func.id if isinstance(func, ast.Name) else getattr(func, "attr", ""))
    return names


def _files(*parts: str) -> list[Path]:
    found: list[Path] = []
    for part in parts:
        target = SRC / part
        if target.is_dir():
            found.extend(target.rglob("*.py"))
        elif target.is_file():
            found.append(target)
    return sorted(found)


def _relative(path: Path) -> str:
    return path.relative_to(SRC).as_posix()


def _importing(files: list[Path], prefix: str) -> list[str]:
    return [_relative(p) for p in files if any(_under(name, prefix) for name in _imports(p))]


def test_every_pure_entry_names_real_files() -> None:
    for part in PURE:
        assert _files(part), part


def test_the_pure_set_imports_no_adapter_service_cli_or_framework() -> None:
    offenders = [
        f"{_relative(path)} imports {banned}"
        for path in _files(*PURE)
        for banned in PURE_BANNED
        if any(_under(name, banned) for name in _imports(path))
    ]
    assert offenders == []


def test_the_pure_set_does_no_io() -> None:
    offenders = [
        f"{_relative(path)}: {name}"
        for path in _files(*PURE)
        for name in sorted(_io_imports(path) | (_calls(path) & IO_CALLS))
    ]
    assert offenders == []


def test_viz_init_re_exports_nothing_from_viz_load() -> None:
    assert _importing(_files("viz/__init__.py"), "hedge_fund.viz.load") == []


def test_services_import_no_adapters() -> None:
    assert _importing(_files("services"), "hedge_fund.adapters") == []


def test_only_the_composition_roots_import_adapters() -> None:
    outside = [p for p in _files(".") if not _relative(p).startswith("adapters/")]
    importers = set(_importing(outside, "hedge_fund.adapters"))
    assert importers <= ADAPTER_IMPORTERS, sorted(importers - ADAPTER_IMPORTERS)
    assert "viz/load.py" in importers


def test_only_cli_imports_the_broker_adapters() -> None:
    outside = [p for p in _files(".") if not _relative(p).startswith("adapters/broker/")]
    importers = set(_importing(outside, "hedge_fund.adapters.broker"))
    assert importers == BROKER_IMPORTERS


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "x.py"
    path.write_text(
        "import subprocess\n"
        "import urllib.request\n"
        "from http.client import HTTPConnection\n"
        "from hedge_fund.adapters import broker\n"
        "from hedge_fund.services.cycle import run_cycle\n"
        "open('f')\n"
        "Path('f').read_text()\n",
        encoding="utf-8",
    )
    return path


def test_the_scanners_see_imports_and_io_calls(source: Path) -> None:
    imports = _imports(source)

    assert {"subprocess", "hedge_fund.adapters", "hedge_fund.adapters.broker"} <= imports
    assert any(_under(name, "hedge_fund.services") for name in imports)
    assert {"open", "read_text"} <= _calls(source)
    assert {"subprocess", "urllib.request", "http.client", "http.client.HTTPConnection"} == (
        _io_imports(source)
    )


def _from_import(text: str) -> ast.ImportFrom:
    node = ast.parse(text).body[0]
    assert isinstance(node, ast.ImportFrom)
    return node


def test_relative_imports_resolve_against_the_package() -> None:
    assert _resolve(SRC / "viz" / "__init__.py", _from_import("from . import load")) == (
        "hedge_fund.viz"
    )
    assert _resolve(SRC / "viz" / "views.py", _from_import("from ..domain import x")) == (
        "hedge_fund.domain"
    )
