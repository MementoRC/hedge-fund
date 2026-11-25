"""Read ``website/mkdocs.yml`` as plain data, without importing mkdocs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

MKDOCS = Path(__file__).resolve().parents[1] / "website" / "mkdocs.yml"


class _TolerantLoader(yaml.SafeLoader):
    """SafeLoader that reads, but does not run, mkdocs' ``!!python/name:`` tags."""


def _python_name(loader: yaml.SafeLoader, suffix: str, node: yaml.Node) -> str:
    return suffix


_TolerantLoader.add_multi_constructor("tag:yaml.org,2002:python/name:", _python_name)


def load_mkdocs_config() -> dict[str, Any]:
    loaded = yaml.load(MKDOCS.read_text(encoding="utf-8"), Loader=_TolerantLoader)  # noqa: S506
    assert isinstance(loaded, dict)
    return loaded
