"""validate_snapshot: each rejection rule, and the reads that must pass."""

from __future__ import annotations

import dataclasses
import math
from datetime import UTC, date, datetime, timedelta, timezone, tzinfo

import pytest

from hedge_fund.domain.account import AccountSnapshot
from hedge_fund.domain.orders import Holding
from hedge_fund.engine.account.validate import validate_snapshot

AS_OF = date(2026, 10, 23)
VALID = AccountSnapshot(
    as_of=AS_OF,
    observed_at=datetime(2026, 10, 23, 13, 0, tzinfo=UTC),
    equity=1_000_000.0,
    holdings=(Holding(ticker="AAPL", shares=40), Holding(ticker="XOM", shares=-25)),
)


class _NoOffset(tzinfo):
    """A tzinfo that knows no offset: ``utcoffset()`` is None, so the time is naive."""

    def utcoffset(self, dt: datetime | None) -> timedelta | None:
        return None


def _holdings(*pairs: tuple[str, object]) -> tuple[Holding, ...]:
    """Holdings built without type checks, so malformed shares reach the validator."""
    return tuple(Holding(ticker=ticker, shares=shares) for ticker, shares in pairs)


@pytest.mark.parametrize(
    "equity", [0.0, 0, -1.0, math.nan, math.inf, -math.inf, True, "100"], ids=repr
)
def test_rejects_equity_that_is_not_a_valid_nav(equity: object) -> None:
    snapshot = dataclasses.replace(VALID, equity=equity)
    with pytest.raises(ValueError, match="nav must be a finite number > 0"):
        validate_snapshot(snapshot)


@pytest.mark.parametrize(
    "observed_at",
    [datetime(2026, 10, 23, 13, 0), datetime(2026, 10, 23, 13, 0, tzinfo=_NoOffset())],
    ids=["no-tzinfo", "tzinfo-without-offset"],
)
def test_rejects_a_naive_observed_at(observed_at: datetime) -> None:
    with pytest.raises(ValueError, match="observed_at must be tz-aware"):
        validate_snapshot(dataclasses.replace(VALID, observed_at=observed_at))


@pytest.mark.parametrize(
    ("holdings", "message"),
    [
        pytest.param(
            _holdings(("XOM", -25), ("AAPL", 40)),
            "strictly ascending by ticker, got 'AAPL' after 'XOM'",
            id="unsorted",
        ),
        pytest.param(
            _holdings(("AAPL", 40), ("AAPL", 5)),
            "strictly ascending by ticker, got 'AAPL' after 'AAPL'",
            id="repeated",
        ),
        pytest.param(_holdings(("AAPL", 0)), "'AAPL' shares must be a non-zero int", id="zero"),
        pytest.param(_holdings(("AAPL", True)), "'AAPL' shares must be a non-zero int", id="bool"),
        pytest.param(_holdings(("AAPL", 1.0)), "'AAPL' shares must be a non-zero int", id="float"),
        pytest.param(_holdings(("AAPL", "3")), "'AAPL' shares must be a non-zero int", id="str"),
    ],
)
def test_rejects_malformed_holdings(holdings: tuple[Holding, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_snapshot(dataclasses.replace(VALID, holdings=holdings))


@pytest.mark.parametrize(
    "snapshot",
    [
        pytest.param(VALID, id="long-and-short"),
        pytest.param(dataclasses.replace(VALID, holdings=()), id="empty-holdings"),
        pytest.param(dataclasses.replace(VALID, equity=250), id="int-equity"),
        pytest.param(
            dataclasses.replace(
                VALID,
                observed_at=datetime(2026, 10, 23, 9, 0, tzinfo=timezone(timedelta(hours=-4))),
            ),
            id="non-utc-offset",
        ),
        pytest.param(
            dataclasses.replace(VALID, observed_at=datetime(2026, 10, 22, 23, 0, tzinfo=UTC)),
            id="read-the-evening-before",
        ),
    ],
)
def test_accepts_valid_reads(snapshot: AccountSnapshot) -> None:
    validate_snapshot(snapshot)
