"""Tests for PriceSnapshot contract validation."""

import dataclasses
from datetime import date

import pandas as pd
import pytest

from hedge_fund.domain.snapshot import PriceSnapshot

DATES = pd.DatetimeIndex(["2026-01-02", "2026-01-05", "2026-01-06"])
SECTORS = {"AAA": "Tech", "BBB": "Energy"}
AS_OF = date(2026, 1, 6)


def _closes(index: pd.Index = DATES) -> pd.DataFrame:
    return pd.DataFrame({"AAA": [1.0, 2.0, 3.0], "BBB": [4.0, 5.0, 6.0]}, index=index)


def test_valid_snapshot_keeps_fields() -> None:
    closes = _closes()
    snap = PriceSnapshot(as_of=AS_OF, closes=closes, sectors=SECTORS)
    assert snap.as_of == AS_OF
    assert snap.closes is closes
    assert snap.sectors == SECTORS


def test_as_of_after_last_row_is_allowed() -> None:
    snap = PriceSnapshot(as_of=date(2026, 3, 1), closes=_closes(), sectors=SECTORS)
    assert snap.as_of == date(2026, 3, 1)


def test_extra_sector_keys_are_allowed() -> None:
    sectors = {**SECTORS, "ZZZ": "Utilities"}
    snap = PriceSnapshot(as_of=AS_OF, closes=_closes(), sectors=sectors)
    assert snap.sectors == sectors


@pytest.mark.parametrize(
    "closes",
    [pd.DataFrame(index=pd.DatetimeIndex([])), pd.DataFrame(index=DATES)],
    ids=["no-rows", "no-columns"],
)
def test_empty_closes_rejected(closes: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="empty"):
        PriceSnapshot(as_of=AS_OF, closes=closes, sectors=SECTORS)


def test_non_datetime_index_rejected() -> None:
    with pytest.raises(ValueError, match="DatetimeIndex"):
        PriceSnapshot(as_of=AS_OF, closes=_closes(pd.Index([0, 1, 2])), sectors=SECTORS)


def test_unsorted_index_rejected() -> None:
    index = pd.DatetimeIndex(["2026-01-05", "2026-01-02", "2026-01-06"])
    with pytest.raises(ValueError, match="sorted"):
        PriceSnapshot(as_of=AS_OF, closes=_closes(index), sectors=SECTORS)


def test_duplicate_dates_rejected() -> None:
    index = pd.DatetimeIndex(["2026-01-02", "2026-01-02", "2026-01-06"])
    with pytest.raises(ValueError, match="duplicate"):
        PriceSnapshot(as_of=AS_OF, closes=_closes(index), sectors=SECTORS)


def test_duplicate_tickers_rejected() -> None:
    closes = pd.DataFrame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], index=DATES, columns=["AAA", "AAA"])
    with pytest.raises(ValueError, match="duplicate tickers"):
        PriceSnapshot(as_of=AS_OF, closes=closes, sectors=SECTORS)


def test_rows_after_as_of_rejected() -> None:
    with pytest.raises(ValueError, match="after as_of"):
        PriceSnapshot(as_of=date(2026, 1, 5), closes=_closes(), sectors=SECTORS)


def test_ticker_without_sector_rejected() -> None:
    with pytest.raises(ValueError, match="BBB"):
        PriceSnapshot(as_of=AS_OF, closes=_closes(), sectors={"AAA": "Tech"})


def test_snapshot_is_frozen() -> None:
    snap = PriceSnapshot(as_of=AS_OF, closes=_closes(), sectors=SECTORS)
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.as_of = date(2026, 2, 1)  # type: ignore[misc]
