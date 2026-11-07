"""Point-in-time membership: boundaries, parsing, composition conversion, universes."""

from __future__ import annotations

from datetime import date

import pytest

from hedge_fund.domain.backtest import MemberSpan
from hedge_fund.engine.backtest.membership import (
    PointInTimeUniverse,
    ever_members,
    format_membership,
    parse_flagged_spans,
    parse_membership,
    spans_from_compositions,
)

SPANS = (
    MemberSpan("AAA", date(2024, 1, 2), None),
    MemberSpan("BBB", date(2024, 1, 2), date(2024, 3, 15)),
    MemberSpan("CCC", date(2024, 3, 15), None),
)
UNIVERSE = PointInTimeUniverse(
    spans=SPANS,
    sectors={"AAA": "Energy", "BBB": "Financials"},
    ciks={"AAA": "0000000001", "CCC": "0000000003"},
)
TEXT = (
    "# source: example @ abc123\n"
    "ticker,start,end\n"
    "BBB,2024-01-02,2024-03-15\n"
    "AAA,2024-01-02,\n"
    "CCC,2024-03-15,\n"
)


def test_the_day_before_a_change_has_the_old_members() -> None:
    assert UNIVERSE.tickers_on(date(2024, 3, 14)) == ("AAA", "BBB")


def test_on_the_effective_date_the_addition_is_in_and_the_removal_is_out() -> None:
    assert UNIVERSE.tickers_on(date(2024, 3, 15)) == ("AAA", "CCC")


def test_before_any_span_there_are_no_members() -> None:
    assert UNIVERSE.tickers_on(date(2024, 1, 1)) == ()


def test_members_builds_a_universe_dated_on_the_day() -> None:
    universe = UNIVERSE.members(date(2024, 3, 15))
    assert universe.name == "sp500_pit"
    assert universe.list_date == date(2024, 3, 15)
    assert universe.tickers == ("AAA", "CCC")
    assert dict(universe.sectors) == {"AAA": "Energy"}
    assert dict(universe.ciks) == {"AAA": "0000000001", "CCC": "0000000003"}


def test_ever_members_are_the_spans_overlapping_the_window() -> None:
    assert ever_members(SPANS, date(2023, 1, 1), date(2024, 1, 1)) == ()
    assert ever_members(SPANS, date(2024, 3, 15), date(2024, 12, 31)) == ("AAA", "CCC")
    assert ever_members(SPANS, date(2024, 3, 14), date(2024, 3, 15)) == ("AAA", "BBB", "CCC")


def test_parse_skips_comments_and_sorts_by_start_then_ticker() -> None:
    assert parse_membership(TEXT) == SPANS


def test_format_round_trips_with_a_provenance_header() -> None:
    text = format_membership(SPANS, ["source: example"])
    assert text.splitlines()[:2] == ["# source: example", "ticker,start,end"]
    assert parse_membership(text) == SPANS


def test_adjacent_spans_of_one_ticker_are_allowed() -> None:
    text = "ticker,start,end\nAAA,2024-01-02,2024-02-01\nAAA,2024-02-01,\n"
    assert len(parse_membership(text)) == 2


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("ticker,from,end\nAAA,2024-01-02,\n", "header"),
        ("ticker,start,end\nAAA,2024-01-02\n", "missing a column"),
        ("ticker,start,end\nAAA,2024-01-02,,x\n", "extra field"),
        ("ticker,start,end\n ,2024-01-02,\n", "empty ticker"),
        ("ticker,start,end\nAAA,02/01/2024,\n", "YYYY-MM-DD"),
        ("ticker,start,end\nAAA,2024-03-01,2024-03-01\n", "not after start"),
        ("ticker,start,end\nAAA,2024-01-02,2024-03-01\nAAA,2024-02-01,\n", "overlapping"),
        ("ticker,start,end\nAAA,2024-01-02,\nAAA,2024-02-01,\n", "overlapping"),
    ],
)
def test_parse_rejects_bad_files(body: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_membership(body)


FLAGGED_TEXT = (
    "# source: example\n"
    "ticker,start,end,reason\n"
    'AAA,2024-02-01,2024-03-01,"other company, not the current one"\n'
    "NOPE,2024-02-01,2024-03-01,never a member\n"
)
FLAGGED = parse_flagged_spans(FLAGGED_TEXT)
FLAGGED_UNIVERSE = PointInTimeUniverse(
    spans=(
        MemberSpan("AAA", date(2024, 1, 2), None),
        MemberSpan("BBB", date(2024, 1, 2), None),
    ),
    sectors={"AAA": "Energy", "BBB": "Energy"},
    ciks={},
    flagged=FLAGGED,
)


def test_a_ticker_inside_a_flagged_span_is_excluded_and_reported() -> None:
    day = date(2024, 2, 15)
    assert FLAGGED_UNIVERSE.tickers_on(day) == ("BBB",)
    assert FLAGGED_UNIVERSE.members(day).tickers == ("BBB",)
    assert FLAGGED_UNIVERSE.excluded_on(day) == ("AAA",)


def test_the_flagged_span_is_half_open() -> None:
    assert FLAGGED_UNIVERSE.excluded_on(date(2024, 2, 1)) == ("AAA",)
    assert FLAGGED_UNIVERSE.excluded_on(date(2024, 1, 31)) == ()
    assert FLAGGED_UNIVERSE.excluded_on(date(2024, 3, 1)) == ()
    assert FLAGGED_UNIVERSE.tickers_on(date(2024, 3, 1)) == ("AAA", "BBB")


def test_outside_a_flagged_span_nothing_is_excluded() -> None:
    assert FLAGGED_UNIVERSE.tickers_on(date(2024, 1, 31)) == ("AAA", "BBB")


def test_a_flagged_ticker_that_is_not_a_member_is_not_reported() -> None:
    assert "NOPE" not in FLAGGED_UNIVERSE.excluded_on(date(2024, 2, 15))


def test_an_unknown_ticker_in_the_flagged_file_is_fine() -> None:
    assert [flagged.span.ticker for flagged in FLAGGED] == ["AAA", "NOPE"]


def test_a_flagged_span_keeps_its_reason() -> None:
    assert FLAGGED[0].reason == "other company, not the current one"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("ticker,start,end\nAAA,2024-02-01,\n", "header must be"),
        ("ticker,start,end,reason\nAAA,2024-02-01,\n", "missing a column"),
        ("ticker,start,end,reason\nAAA,2024-02-01,,\n", "empty reason"),
        ("ticker,start,end,reason\nAAA,02/01/2024,,x\n", "YYYY-MM-DD"),
        ("ticker,start,end,reason\nAAA,2024-03-01,2024-02-01,x\n", "is not after start"),
    ],
)
def test_a_malformed_flagged_row_fails_closed(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_flagged_spans(text)


def test_compositions_become_spans_from_each_list_to_the_next() -> None:
    rows = [
        (date(2024, 1, 2), frozenset({"AAA", "BBB"})),
        (date(2024, 3, 15), frozenset({"AAA", "CCC"})),
    ]
    assert spans_from_compositions(rows) == SPANS


def test_a_name_that_comes_back_gets_a_second_span() -> None:
    rows = [
        (date(2024, 1, 2), frozenset({"AAA"})),
        (date(2024, 2, 1), frozenset({"BBB"})),
        (date(2024, 3, 1), frozenset({"AAA", "BBB"})),
    ]
    assert spans_from_compositions(rows) == (
        MemberSpan("AAA", date(2024, 1, 2), date(2024, 2, 1)),
        MemberSpan("BBB", date(2024, 2, 1), None),
        MemberSpan("AAA", date(2024, 3, 1), None),
    )


def test_composition_dates_must_increase() -> None:
    rows = [(date(2024, 3, 1), frozenset({"AAA"})), (date(2024, 3, 1), frozenset({"BBB"}))]
    with pytest.raises(ValueError, match="strictly increasing"):
        spans_from_compositions(rows)
