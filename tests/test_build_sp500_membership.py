"""Tests for the pure helpers of scripts/build_sp500_membership.py (the script needs network)."""

from datetime import date

from scripts.build_sp500_membership import compositions, to_yahoo

SOURCE = 'date,tickers\n2024-01-02,"AAPL,BRK.B,MSFT"\n2023-12-18,"AAPL,BRK.B,XYZ"\n'


def test_to_yahoo_uses_dashes_for_share_classes() -> None:
    assert to_yahoo(" brk.b ") == "BRK-B"


def test_compositions_are_sorted_by_date_with_yahoo_tickers() -> None:
    assert compositions(SOURCE) == [
        (date(2023, 12, 18), frozenset({"AAPL", "BRK-B", "XYZ"})),
        (date(2024, 1, 2), frozenset({"AAPL", "BRK-B", "MSFT"})),
    ]


def test_blank_list_entries_are_skipped() -> None:
    assert compositions('date,tickers\n2024-01-02,"AAPL,,MSFT,"\n') == [
        (date(2024, 1, 2), frozenset({"AAPL", "MSFT"}))
    ]
