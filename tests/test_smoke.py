"""Smoke test."""

import hedge_fund


def test_version_present() -> None:
    assert hedge_fund.__version__
