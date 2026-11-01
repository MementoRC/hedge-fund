"""Tests for the research JSON store."""

import dataclasses
import hashlib
import json
import math
import re
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from hedge_fund.adapters.research.store import (
    JsonResearchStore,
    decode_candidate,
    decode_name,
    decode_run,
    decode_usage,
    load_research,
    save_research,
)
from hedge_fund.domain.cost import ModelUsage
from hedge_fund.domain.research import (
    NameResearch,
    Rating,
    ResearchResult,
    RunRecord,
    Verdict,
)
from hedge_fund.domain.shortlist import Candidate, Side
from hedge_fund.ports.research import ResearchStore

AS_OF = date(2026, 9, 30)
USAGE = (
    ModelUsage("claude-opus-5-5", 2, 1200, 300, 100, 400),
    ModelUsage("claude-sonnet-5", 5, 900, 80, 0, 600),
)


def _result(seconds: float = 1.25) -> ResearchResult:
    long = Candidate(ticker="AAPL", sector="Tech", side=Side.LONG, depth=1, composite=1.5)
    short = Candidate(ticker="XOM", sector="Energy", side=Side.SHORT, depth=1, composite=-0.75)
    ok = RunRecord(
        ticker="AAPL",
        as_of=AS_OF,
        run_index=0,
        rating=Rating.BUY,
        decision="buy it",
        reports={"market_report": "m"},
        seconds=seconds,
        error=None,
    )
    failed = RunRecord(
        ticker="AAPL",
        as_of=AS_OF,
        run_index=1,
        rating=None,
        decision="",
        reports={},
        seconds=0.0,
        error="RuntimeError: boom",
    )
    review = RunRecord(
        ticker="XOM",
        as_of=AS_OF,
        run_index=0,
        rating=Rating.REVIEW,
        decision="unclear",
        reports={"news_report": "n"},
        seconds=2.5,
        error=None,
    )
    return ResearchResult(
        as_of=AS_OF,
        names=(
            NameResearch(
                candidate=long,
                runs=(ok, failed),
                valid=1,
                mean=2.0,
                std=0.0,
                agreement=1.0,
                verdict=Verdict.INSUFFICIENT,
            ),
            NameResearch(
                candidate=short,
                runs=(review,),
                valid=0,
                mean=None,
                std=None,
                agreement=None,
                verdict=Verdict.INSUFFICIENT,
            ),
        ),
    )


def test_save_names_the_file_by_as_of_and_digest(tmp_path: Path) -> None:
    path = save_research(_result(), tmp_path)
    assert path.parent == tmp_path / "research"
    assert re.fullmatch(r"2026-09-30_[0-9a-f]{12}\.json", path.name)
    body = json.loads(path.read_text(encoding="utf-8"))
    assert datetime.fromisoformat(body["created_at"]).tzinfo is UTC
    assert body["result"]["as_of"] == "2026-09-30"
    assert body["result"]["names"][0]["candidate"]["side"] == "long"
    assert body["result"]["names"][0]["runs"][0]["rating"] == "Buy"
    assert body["result"]["names"][1]["mean"] is None


def test_round_trip(tmp_path: Path) -> None:
    result = _result()
    assert load_research(save_research(result, tmp_path)) == result


def test_save_is_idempotent_and_returns_existing_file_as_is(tmp_path: Path) -> None:
    first = save_research(_result(), tmp_path)
    text = first.read_text(encoding="utf-8")
    second = save_research(_result(), tmp_path)
    assert second == first
    assert second.read_text(encoding="utf-8") == text
    assert len(list((tmp_path / "research").iterdir())) == 1


def test_different_content_gets_a_different_path(tmp_path: Path) -> None:
    other = dataclasses.replace(_result(), names=_result().names[:1])
    assert save_research(other, tmp_path) != save_research(_result(), tmp_path)


def test_created_at_is_outside_the_digest(tmp_path: Path) -> None:
    result = _result()
    path = save_research(result, tmp_path)
    body = json.loads(path.read_text(encoding="utf-8"))
    body["created_at"] = "1999-01-01T00:00:00+00:00"
    path.write_text(json.dumps(body), encoding="utf-8")
    assert load_research(path) == result


def test_non_finite_floats_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Out of range float"):
        save_research(_result(seconds=float("nan")), tmp_path)


def test_load_rejects_tampered_content(tmp_path: Path) -> None:
    path = save_research(_result(), tmp_path)
    body = json.loads(path.read_text(encoding="utf-8"))
    body["result"]["names"][0]["runs"][0]["decision"] = "tampered"
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_research(path)


def test_load_rejects_stem_without_underscore(tmp_path: Path) -> None:
    path = save_research(_result(), tmp_path)
    renamed = path.rename(path.with_name("2026-09-30.json"))
    with pytest.raises(ValueError, match=re.escape("expected <as_of>_<digest>.json")):
        load_research(renamed)


def test_load_rejects_body_without_result(tmp_path: Path) -> None:
    path = save_research(_result(), tmp_path)
    path.write_text(json.dumps({"created_at": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="malformed research file"):
        load_research(path)


def test_load_rejects_invalid_json(tmp_path: Path) -> None:
    path = save_research(_result(), tmp_path)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed research file"):
        load_research(path)


def test_load_rejects_wrong_types(tmp_path: Path) -> None:
    path = save_research(_result(), tmp_path)
    path.write_text(json.dumps([1, 2]), encoding="utf-8")
    with pytest.raises(ValueError, match="malformed research file"):
        load_research(path)


def test_load_rejects_well_digested_but_malformed_names(tmp_path: Path) -> None:
    path = save_research(_result(), tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))["result"]
    del payload["names"][0]["verdict"]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    forged = path.with_name(f"2026-09-30_{digest}.json")
    forged.write_text(json.dumps({"result": payload}), encoding="utf-8")
    with pytest.raises(ValueError, match="malformed research file") as info:
        load_research(forged)
    assert isinstance(info.value.__cause__, KeyError)


def test_load_rejects_as_of_mismatch(tmp_path: Path) -> None:
    path = save_research(_result(), tmp_path)
    renamed = path.rename(path.with_name(path.name.replace("2026-09-30", "2026-09-29")))
    with pytest.raises(ValueError, match="as_of mismatch"):
        load_research(renamed)


# --- JsonResearchStore ------------------------------------------------------------------


def _set_created_at(path: Path, value: object) -> None:
    body = json.loads(path.read_text(encoding="utf-8"))
    body["created_at"] = value
    path.write_text(json.dumps(body), encoding="utf-8")


def test_json_store_satisfies_the_port_and_round_trips(tmp_path: Path) -> None:
    store: ResearchStore = JsonResearchStore(tmp_path)
    store.save(_result())
    assert store.load(AS_OF) == _result()
    [path] = (tmp_path / "research").iterdir()
    assert re.fullmatch(r"2026-09-30_[0-9a-f]{12}\.json", path.name)


@pytest.mark.parametrize("setup", ["missing-dir", "other-date", "empty-dir"])
def test_json_store_load_is_none_when_nothing_matches(tmp_path: Path, setup: str) -> None:
    if setup == "other-date":
        save_research(dataclasses.replace(_result(), as_of=date(2026, 9, 29)), tmp_path)
    if setup == "empty-dir":
        (tmp_path / "research").mkdir()
    assert JsonResearchStore(tmp_path).load(AS_OF) is None


def test_json_store_newest_created_at_wins(tmp_path: Path) -> None:
    first, second = _result(seconds=1.0), _result(seconds=2.0)
    first_path, second_path = save_research(first, tmp_path), save_research(second, tmp_path)
    _set_created_at(first_path, "2026-09-30T23:00:00+00:00")
    _set_created_at(second_path, "2026-09-30T22:00:00+00:00")
    assert JsonResearchStore(tmp_path).load(AS_OF) == first
    _set_created_at(second_path, "2026-09-30T19:30:00-04:00")  # 23:30 UTC: newer by instant
    assert JsonResearchStore(tmp_path).load(AS_OF) == second


def test_json_store_created_at_tie_goes_to_the_greater_file_name(tmp_path: Path) -> None:
    paths = {seconds: save_research(_result(seconds=seconds), tmp_path) for seconds in (1.0, 2.0)}
    for path in paths.values():
        _set_created_at(path, "2026-09-30T22:00:00+00:00")
    greater = max(paths, key=lambda seconds: paths[seconds].name)
    assert JsonResearchStore(tmp_path).load(AS_OF) == _result(seconds=greater)


def test_json_store_rejects_a_tampered_file(tmp_path: Path) -> None:
    path = save_research(_result(), tmp_path)
    body = json.loads(path.read_text(encoding="utf-8"))
    body["result"]["names"][0]["runs"][0]["decision"] = "tampered"
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        JsonResearchStore(tmp_path).load(AS_OF)


@pytest.mark.parametrize(
    "created_at",
    [
        pytest.param(None, id="null"),
        pytest.param(20260930, id="int"),
        pytest.param("yesterday", id="unparseable"),
        pytest.param("2026-09-30T22:00:00", id="naive"),
    ],
)
def test_json_store_rejects_a_bad_created_at(tmp_path: Path, created_at: object) -> None:
    path = save_research(_result(), tmp_path)
    _set_created_at(path, created_at)
    with pytest.raises(ValueError, match=f"{re.escape(path.name)}: malformed created_at"):
        JsonResearchStore(tmp_path).load(AS_OF)


def test_json_store_rejects_a_missing_created_at(tmp_path: Path) -> None:
    path = save_research(_result(), tmp_path)
    body = json.loads(path.read_text(encoding="utf-8"))
    del body["created_at"]
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match=f"{re.escape(path.name)}: malformed created_at"):
        JsonResearchStore(tmp_path).load(AS_OF)


# --- public decoders --------------------------------------------------------------------


def test_decode_candidate() -> None:
    item = {"ticker": "AAPL", "sector": "Tech", "side": "short", "depth": 2, "composite": -1}
    assert decode_candidate(item) == Candidate("AAPL", "Tech", Side.SHORT, 2, -1.0)


def test_decoders_keep_null_stats() -> None:
    name = _result().names[1]
    payload = json.loads(json.dumps(dataclasses.asdict(name), default=str))
    assert decode_name(payload) == name


def test_decoders_turn_non_finite_strings_back() -> None:
    run = {
        "ticker": "AAPL",
        "as_of": "2026-09-30",
        "run_index": 0,
        "rating": None,
        "decision": "",
        "reports": {},
        "seconds": "inf",
        "error": "RuntimeError: x",
    }
    item = {
        "candidate": {
            "ticker": "AAPL",
            "sector": "Tech",
            "side": "long",
            "depth": 1,
            "composite": "-inf",
        },
        "runs": [run],
        "valid": 0,
        "mean": "nan",
        "std": "inf",
        "agreement": "nan",
        "verdict": "insufficient",
    }
    name = decode_name(item)
    assert name.candidate.composite == -math.inf
    assert decode_run(run).seconds == math.inf
    assert name.mean is not None
    assert math.isnan(name.mean)
    assert name.std == math.inf
    assert name.agreement is not None
    assert math.isnan(name.agreement)


# --- dates ---------------------------------------------------------------------------------


def test_json_store_dates_are_sorted_and_unique(tmp_path: Path) -> None:
    for as_of, seconds in [
        (date(2026, 9, 30), 1.0),
        (date(2026, 9, 1), 1.0),
        (date(2026, 9, 30), 2.0),  # a second result for the same day: another file
    ]:
        save_research(dataclasses.replace(_result(seconds=seconds), as_of=as_of), tmp_path)
    assert len(list((tmp_path / "research").iterdir())) == 3
    store: ResearchStore = JsonResearchStore(tmp_path)
    assert store.dates() == (date(2026, 9, 1), date(2026, 9, 30))


@pytest.mark.parametrize(
    "name",
    [
        "notes.txt",
        "2026-09-30.json",
        "2026-09-30_abc.json",
        "2026-09-30_0123456789AB.json",
        "2026-13-45_0123456789ab.json",
        "x2026-09-30_0123456789ab.json",
        "2026-09-30_0123456789ab.json.tmp",
    ],
)
def test_json_store_dates_ignore_names_that_do_not_match(tmp_path: Path, name: str) -> None:
    save_research(dataclasses.replace(_result(), as_of=date(2026, 9, 29)), tmp_path)
    (tmp_path / "research" / name).write_text("{}", encoding="utf-8")
    assert JsonResearchStore(tmp_path).dates() == (date(2026, 9, 29),)


def test_json_store_dates_ignore_directories_named_like_a_file(tmp_path: Path) -> None:
    save_research(dataclasses.replace(_result(), as_of=date(2026, 9, 29)), tmp_path)
    (tmp_path / "research" / "2026-09-30_0123456789ab.json").mkdir()
    assert JsonResearchStore(tmp_path).dates() == (date(2026, 9, 29),)


@pytest.mark.parametrize("setup", ["missing-dir", "empty-dir"])
def test_json_store_dates_are_empty_without_files(tmp_path: Path, setup: str) -> None:
    if setup == "empty-dir":
        (tmp_path / "research").mkdir()
    assert JsonResearchStore(tmp_path).dates() == ()


def test_json_store_dates_do_not_read_the_files(tmp_path: Path) -> None:
    (tmp_path / "research").mkdir()
    (tmp_path / "research" / "2026-09-30_0123456789ab.json").write_text("not json")
    store = JsonResearchStore(tmp_path)
    assert store.dates() == (date(2026, 9, 30),)
    with pytest.raises(ValueError, match="malformed research file"):
        store.load(date(2026, 9, 30))


# --- usage ---------------------------------------------------------------------------------


def _with_usage() -> ResearchResult:
    """``_result()`` whose first run carries USAGE (the errored second run has none)."""
    first = _result().names[0]
    ok = dataclasses.replace(first.runs[0], usage=USAGE)
    name = dataclasses.replace(first, runs=(ok, first.runs[1]))
    return dataclasses.replace(_result(), names=(name, *_result().names[1:]))


def test_usage_is_written_as_a_list_or_null_and_round_trips(tmp_path: Path) -> None:
    result = _with_usage()
    path = save_research(result, tmp_path)
    runs = json.loads(path.read_text(encoding="utf-8"))["result"]["names"][0]["runs"]
    assert runs[0]["usage"][0] == {
        "model": "claude-opus-5-5",
        "calls": 2,
        "input_tokens": 1200,
        "output_tokens": 300,
        "cache_write_tokens": 100,
        "cache_read_tokens": 400,
    }
    assert runs[1]["usage"] is None
    assert load_research(path) == result


def test_an_old_file_without_usage_still_verifies_and_loads(tmp_path: Path) -> None:
    path = save_research(_result(), tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))["result"]
    for name in payload["names"]:
        for run in name["runs"]:
            del run["usage"]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    old = path.with_name(f"2026-09-30_{digest}.json")
    body = {"created_at": "2026-09-30T22:00:00+00:00", "result": payload}
    old.write_text(json.dumps(body), encoding="utf-8")
    assert old != path  # the new format's digest differs: only dedup notices
    assert load_research(old) == _result()


def _break_row(usage: list[dict[str, object]]) -> object:
    del usage[0]["cache_read_tokens"]
    return usage


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(_break_row, id="row-missing-a-key"),
        pytest.param(lambda _usage: "lots", id="string"),
        pytest.param(lambda _usage: {"model": "x"}, id="dict"),
        pytest.param(lambda _usage: 7, id="int"),
    ],
)
def test_a_malformed_usage_value_fails_the_load_as_a_malformed_file(
    tmp_path: Path, bad: Callable[[list[dict[str, object]]], object]
) -> None:
    path = save_research(_with_usage(), tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))["result"]
    run = payload["names"][0]["runs"][0]
    run["usage"] = bad(run["usage"])
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    forged = path.with_name(f"2026-09-30_{digest}.json")
    forged.write_text(json.dumps({"result": payload}), encoding="utf-8")
    with pytest.raises(ValueError, match="malformed research file"):
        load_research(forged)


def test_decode_usage() -> None:
    item = {
        "model": "claude-sonnet-5",
        "calls": 5,
        "input_tokens": 900,
        "output_tokens": 80,
        "cache_write_tokens": 0,
        "cache_read_tokens": 600,
    }
    assert decode_usage(item) == USAGE[1]
