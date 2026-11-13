"""The MkDocs skeleton: valid config, nav files that exist, pages shaped as promised.

mkdocs is not in the dev environment, so nothing here imports it: the config is parsed as YAML.
The real ``mkdocs build --strict`` runs in the ``site`` environment and the Pages workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from hedge_fund.domain.backtest import RESIDUAL_BIASES

ROOT = Path(__file__).resolve().parents[1]
WEBSITE = ROOT / "website"
MKDOCS = WEBSITE / "mkdocs.yml"
DOCS = WEBSITE / "docs"


class _TolerantLoader(yaml.SafeLoader):
    """SafeLoader that reads, but does not run, mkdocs' ``!!python/name:`` tags."""


def _python_name(loader: yaml.SafeLoader, suffix: str, node: yaml.Node) -> str:
    return suffix


_TolerantLoader.add_multi_constructor("tag:yaml.org,2002:python/name:", _python_name)


def _config() -> dict[str, Any]:
    loaded = yaml.load(MKDOCS.read_text(encoding="utf-8"), Loader=_TolerantLoader)  # noqa: S506
    assert isinstance(loaded, dict)
    return loaded


def _nav_files(nav: list[Any]) -> list[str]:
    files: list[str] = []
    for item in nav:
        if isinstance(item, str):
            files.append(item)
            continue
        for value in item.values():
            files.extend(_nav_files(value) if isinstance(value, list) else [value])
    return files


def test_mkdocs_yml_is_valid_yaml_with_a_site_name() -> None:
    assert _config()["site_name"] == "hedge-fund"


def test_the_theme_is_material_with_a_light_and_dark_toggle() -> None:
    theme = _config()["theme"]

    assert theme["name"] == "material"
    schemes = [entry["scheme"] for entry in theme["palette"]]
    assert schemes == ["default", "slate"]
    assert all("toggle" in entry for entry in theme["palette"])


def test_mermaid_is_a_custom_superfence() -> None:
    extensions = _config()["markdown_extensions"]
    fences = next(e["pymdownx.superfences"] for e in extensions if isinstance(e, dict))

    assert fences["custom_fences"][0]["name"] == "mermaid"
    assert fences["custom_fences"][0]["class"] == "mermaid"


def test_every_nav_entry_points_at_an_existing_file() -> None:
    files = _nav_files(_config()["nav"])

    assert files
    assert [name for name in files if not (DOCS / name).is_file()] == []


def test_every_nav_page_starts_with_a_single_h1() -> None:
    for name in _nav_files(_config()["nav"]):
        lines = (DOCS / name).read_text(encoding="utf-8").splitlines()
        h1s = [line for line in lines if line.startswith("# ")]
        assert lines[0].startswith("# "), name
        assert len(h1s) == 1, name


def test_the_generated_directory_is_gitignored() -> None:
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "website/docs/generated/" in ignored


def test_the_nav_lists_the_five_prose_pages_in_order() -> None:
    assert _nav_files(_config()["nav"]) == [
        "index.md",
        "architecture.md",
        "cycle.md",
        "design-rules.md",
        "backtest-methodology.md",
    ]


def test_the_methodology_page_bias_list_equals_the_residual_biases() -> None:
    page = (DOCS / "backtest-methodology.md").read_text(encoding="utf-8")
    section = page.split("## Residual biases", 1)[1].split("\n## ", 1)[0]
    bullets = [line[2:] for line in section.splitlines() if line.startswith("- ")]

    assert bullets == list(RESIDUAL_BIASES)
