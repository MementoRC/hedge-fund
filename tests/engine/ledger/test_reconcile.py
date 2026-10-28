"""reconcile: per-ticker lot totals against broker positions, in canonical symbols."""

from __future__ import annotations

from datetime import UTC, date, datetime

from hedge_fund.domain.ledger import Fill, FillStatus, Lot, ReconciliationBreak
from hedge_fund.domain.orders import Holding, OrderAction
from hedge_fund.engine.ledger.lots import build_lots
from hedge_fund.engine.ledger.reconcile import reconcile

AT = datetime(2026, 10, 1, 13, 30, tzinfo=UTC)


def _lot(ticker: str, shares: int) -> Lot:
    return Lot(ticker=ticker, shares=shares, entry_price=10.0, entered_at=AT, source="x")


def _held(ticker: str, shares: int) -> Holding:
    return Holding(ticker=ticker, shares=shares)


def test_matching_totals_are_not_breaks() -> None:
    lots = (_lot("AAA", 5), _lot("AAA", 5), _lot("BBB", -3))
    assert reconcile(lots, (_held("AAA", 10), _held("BBB", -3)), exclude=()) == ()


def test_a_differing_count_is_a_break() -> None:
    assert reconcile((_lot("AAA", 10),), (_held("AAA", 12),), exclude=()) == (
        ReconciliationBreak(ticker="AAA", ledger_shares=10, broker_shares=12),
    )


def test_a_broker_only_ticker_is_a_break() -> None:
    assert reconcile((), (_held("CCC", 4),), exclude=()) == (ReconciliationBreak("CCC", 0, 4),)


def test_a_ledger_only_ticker_is_a_break() -> None:
    assert reconcile((_lot("DDD", 7),), (), exclude=()) == (ReconciliationBreak("DDD", 7, 0),)


def test_lots_netting_to_zero_without_a_position_are_not_a_break() -> None:
    assert reconcile((_lot("AAA", 5), _lot("AAA", -5)), (), exclude=()) == ()


def test_excluded_tickers_are_skipped() -> None:
    lots = (_lot("AAA", 10),)
    positions = (_held("AAA", 12), _held("BBB", 3))
    assert reconcile(lots, positions, exclude={"AAA", "BBB"}) == ()
    assert reconcile(lots, positions, exclude={"AAA"}) == (ReconciliationBreak("BBB", 0, 3),)


def test_exclusion_compares_canonical_symbols() -> None:
    assert reconcile((_lot("BRK-B", 10),), (_held("BRK.B", 12),), exclude={"brk.b"}) == ()


def test_a_dot_position_reconciles_against_dash_lots() -> None:
    assert reconcile((_lot("BRK-B", 10),), (_held("BRK.B", 10),), exclude=()) == ()
    assert reconcile((_lot("BRK-B", 10),), (_held("BRK.B", 11),), exclude=()) == (
        ReconciliationBreak("BRK-B", 10, 11),
    )


def test_breaks_are_sorted_by_ticker() -> None:
    lots = (_lot("ZZZ", 1), _lot("AAA", 1))
    assert [b.ticker for b in reconcile(lots, (_held("MMM", 1),), exclude=())] == [
        "AAA",
        "MMM",
        "ZZZ",
    ]


def test_a_position_whose_only_fill_is_not_submitted_is_a_break() -> None:
    never = Fill(
        client_order_id="hf-2026-09-30-AAA-buy-10",
        broker_order_id=None,
        cycle_as_of=date(2026, 9, 30),
        ticker="AAA",
        action=OrderAction.BUY,
        requested_shares=10,
        filled_shares=0,
        avg_price=None,
        filled_at=None,
        status=FillStatus.NOT_SUBMITTED,
        recorded_at=AT,
    )
    lots, _ = build_lots((), (never,))
    assert reconcile(lots, (_held("AAA", 10),), exclude=()) == (ReconciliationBreak("AAA", 0, 10),)
