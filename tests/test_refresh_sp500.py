"""Tests for the pure CIK-join helpers of scripts/refresh_sp500.py (the script needs network)."""

from scripts.refresh_sp500 import join_ciks, sec_ciks, too_many_missing

SEC_PAYLOAD = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 1067983, "ticker": "BRK-B", "title": "BERKSHIRE HATHAWAY INC"},
    "2": {"cik_str": 1652044, "ticker": "GOOGL", "title": "Alphabet Inc."},
    "3": {"cik_str": 1652044, "ticker": "GOOG", "title": "Alphabet Inc."},
    "4": {"cik_str": 999, "ticker": "AAPL", "title": "Later duplicate ticker"},
}


def test_sec_ciks_zero_pads_to_ten_digits_and_keeps_first_entry() -> None:
    ciks = sec_ciks(SEC_PAYLOAD)
    assert ciks["AAPL"] == "0000320193"
    assert ciks["BRK-B"] == "0001067983"
    assert ciks["GOOG"] == ciks["GOOGL"] == "0001652044"


def test_join_ciks_matches_exact_tickers_and_lists_missing_sorted() -> None:
    found, missing = join_ciks(["ZZZZ", "AAPL", "BRK.B", "BRK-B"], sec_ciks(SEC_PAYLOAD))
    assert found == {"AAPL": "0000320193", "BRK-B": "0001067983"}
    assert missing == ["BRK.B", "ZZZZ"]


def test_too_many_missing_allows_up_to_two_percent() -> None:
    assert not too_many_missing(10, 500)
    assert too_many_missing(11, 500)
