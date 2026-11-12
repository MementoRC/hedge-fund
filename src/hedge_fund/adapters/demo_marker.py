"""``<root>/DEMO.json``: the file that makes a root a demo root, written last."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from hedge_fund.adapters.data._atomic import atomic_create
from hedge_fund.domain.demo import DemoMarker

DEMO_MARKER = "DEMO.json"
MARKER_FORMAT = 1
_KEYS = frozenset({"kind", "format", "seed", "store_digest", "run_id"})


def demo_store_dir(root: Path) -> Path:
    """The frozen store of a demo root: the sibling ``<root>.store``."""
    return root.with_name(f"{root.name}.store")


def write_demo_marker(root: Path, marker: DemoMarker) -> Path:
    """Write the marker atomically; FileExistsError if there is one."""
    path = root / DEMO_MARKER
    body = json.dumps(
        {
            "kind": "demo",
            "format": MARKER_FORMAT,
            "seed": marker.seed,
            "store_digest": marker.store_digest,
            "run_id": marker.run_id,
        },
        indent=2,
    )
    atomic_create(path, lambda tmp: tmp.write_text(body, encoding="utf-8"))
    return path


def read_demo_marker(root: Path) -> DemoMarker | None:
    """The marker of ``root``; None only when nothing exists at ``DEMO.json``.

    Fails closed: raises ValueError ("not a demo marker") for anything there that is not a
    regular file (a directory, a dangling symlink), that cannot be examined, that does not
    parse, or that is not exactly a format-1 demo marker.
    """
    path = root / DEMO_MARKER
    try:
        try:
            os.lstat(path)  # unlike os.path.lexists, a PermissionError is not "absent"
        except FileNotFoundError:
            return None
        if not path.is_file():
            raise ValueError("not a regular file")
        return _decode(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"{path}: not a demo marker: {exc}") from exc


def _decode(obj: Any) -> DemoMarker:
    if not isinstance(obj, dict) or set(obj) != _KEYS:
        raise ValueError(f"expected an object with exactly the keys {sorted(_KEYS)}")
    if obj["kind"] != "demo" or obj["format"] != MARKER_FORMAT:
        raise ValueError(f"kind {obj['kind']!r} format {obj['format']!r}")
    seed, digest, run_id = obj["seed"], obj["store_digest"], obj["run_id"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed {seed!r} is not an int")
    if not isinstance(digest, str) or not isinstance(run_id, str):
        raise TypeError("store_digest and run_id must be strings")
    return DemoMarker(seed=seed, store_digest=digest, run_id=run_id)
