"""Tests for the research JSON store."""

import dataclasses
import hashlib
import json
import re
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from hedge_fund.adapters.research.store import load_research, save_research
from hedge_fund.domain.research import (
    NameResearch,
    Rating,
    ResearchResult,
    RunRecord,
    Verdict,
)
from hedge_fund.domain.shortlist import Candidate, Side

AS_OF = date(2026, 9, 30)


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
