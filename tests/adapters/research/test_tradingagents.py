"""Tests for the TradingAgents-backed researcher (offline: stub graph, fake modules)."""

import dataclasses
import sys
import types
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from hedge_fund.adapters.research import tradingagents as ta
from hedge_fund.adapters.research.tradingagents import ResearchConfig, TradingAgentsResearcher
from hedge_fund.domain.research import Rating

AS_OF = date(2026, 9, 30)
ANALYSTS = ("market", "social", "news", "fundamentals")
BASE = {"llm_provider": "openai", "results_dir": "/base/results", "unrelated": "kept"}
FULL_STATE = {
    "market_report": "M",
    "sentiment_report": "S",
    "news_report": "N",
    "fundamentals_report": "F",
    "final_trade_decision": "FINAL: buy",
}


def _install(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture(autouse=True)
def fake_default_config(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, "tradingagents.default_config", build_default_config=lambda: dict(BASE))


class StubFactory:
    """Graph factory that is also the graph: records config and what it saw at propagate time."""

    def __init__(
        self,
        state: dict[str, Any] | None = None,
        signal: str = "Buy",
        error: Exception | None = None,
    ) -> None:
        self.state = FULL_STATE if state is None else state
        self.signal = signal
        self.error = error
        self.config: dict[str, Any] = {}
        self.analysts: Sequence[str] = ()
        self.propagate_calls: list[tuple[str, str]] = []
        self.work_dir: Path | None = None
        self.work_existed = False
        self.memory_dir_existed = True

    def __call__(self, config: dict[str, Any], analysts: Sequence[str]) -> "StubFactory":
        self.config = config
        self.analysts = analysts
        return self

    def propagate(self, ticker: str, date_str: str) -> tuple[dict[str, Any], str]:
        self.propagate_calls.append((ticker, date_str))
        self.work_dir = Path(self.config["results_dir"]).parent
        self.work_existed = self.work_dir.is_dir()
        self.memory_dir_existed = (self.work_dir / "memory").exists()
        if self.error is not None:
            raise self.error
        return self.state, self.signal


def _researcher(root: Path, factory: StubFactory) -> TradingAgentsResearcher:
    return TradingAgentsResearcher(ResearchConfig(root=root), factory)


def test_research_config_defaults_and_frozen(tmp_path: Path) -> None:
    config = ResearchConfig(root=tmp_path)
    assert config.provider == "anthropic"
    assert config.deep_model == "claude-opus-5-5"
    assert config.quick_model == "claude-sonnet-5"
    assert config.debate_rounds == 1
    assert config.risk_rounds == 1
    assert config.analysts == ANALYSTS
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.provider = "openai"  # type: ignore[misc]


def test_run_returns_a_record(tmp_path: Path) -> None:
    factory = StubFactory()
    record = _researcher(tmp_path / "nested" / "research", factory).run("AAPL", AS_OF, 2)
    assert record.ticker == "AAPL"
    assert record.as_of == AS_OF
    assert record.run_index == 2
    assert record.rating is Rating.BUY
    assert record.decision == "FINAL: buy"
    assert record.reports == {
        "market_report": "M",
        "sentiment_report": "S",
        "news_report": "N",
        "fundamentals_report": "F",
    }
    assert record.seconds >= 0.0
    assert record.error is None
    assert record.valid
    assert factory.propagate_calls == [("AAPL", "2026-09-30")]


def test_graph_config_keys_and_temp_layout(tmp_path: Path) -> None:
    root = tmp_path / "nested" / "research"
    factory = StubFactory()
    _researcher(root, factory).run("AAPL", AS_OF, 0)
    config = factory.config
    work = factory.work_dir
    assert work is not None
    assert work.parent == root
    assert factory.analysts == ANALYSTS
    assert config["llm_provider"] == "anthropic"  # explicit values win over the base config
    assert config["deep_think_llm"] == "claude-opus-5-5"
    assert config["quick_think_llm"] == "claude-sonnet-5"
    assert config["max_debate_rounds"] == 1
    assert config["max_risk_discuss_rounds"] == 1
    assert config["results_dir"] == str(work / "results")
    assert config["data_cache_dir"] == str(work / "cache")
    assert config["memory_log_path"] == str(work / "memory" / "trading_memory.md")
    assert config["unrelated"] == "kept"  # base config is the starting point
    assert factory.work_existed
    assert not factory.memory_dir_existed  # path only, not pre-created


def test_custom_config_values_are_passed(tmp_path: Path) -> None:
    factory = StubFactory()
    config = ResearchConfig(
        root=tmp_path,
        provider="openai",
        deep_model="deep",
        quick_model="quick",
        debate_rounds=3,
        risk_rounds=2,
        analysts=("market",),
    )
    TradingAgentsResearcher(config, factory).run("AAPL", AS_OF, 0)
    assert factory.analysts == ("market",)
    assert factory.config["llm_provider"] == "openai"
    assert factory.config["deep_think_llm"] == "deep"
    assert factory.config["quick_think_llm"] == "quick"
    assert factory.config["max_debate_rounds"] == 3
    assert factory.config["max_risk_discuss_rounds"] == 2


def test_temp_dir_removed_after_success(tmp_path: Path) -> None:
    _researcher(tmp_path / "r", StubFactory()).run("AAPL", AS_OF, 0)
    assert list((tmp_path / "r").iterdir()) == []


def test_temp_dir_removed_even_on_exception(tmp_path: Path) -> None:
    factory = StubFactory(error=RuntimeError("graph blew up"))
    with pytest.raises(RuntimeError, match="graph blew up"):
        _researcher(tmp_path / "r", factory).run("AAPL", AS_OF, 0)
    assert list((tmp_path / "r").iterdir()) == []


@pytest.mark.parametrize(
    ("signal", "expected"),
    [
        ("Buy", Rating.BUY),
        ("Overweight", Rating.OVERWEIGHT),
        ("Hold", Rating.HOLD),
        ("Underweight", Rating.UNDERWEIGHT),
        ("Sell", Rating.SELL),
        ("REVIEW", Rating.REVIEW),
    ],
)
def test_signal_maps_to_rating(tmp_path: Path, signal: str, expected: Rating) -> None:
    record = _researcher(tmp_path, StubFactory(signal=signal)).run("AAPL", AS_OF, 0)
    assert record.rating is expected


def test_unknown_signal_is_a_value_error_and_cleans_up(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Maybe"):
        _researcher(tmp_path / "r", StubFactory(signal="Maybe")).run("AAPL", AS_OF, 0)
    assert list((tmp_path / "r").iterdir()) == []


def test_missing_reports_default_to_empty_strings(tmp_path: Path) -> None:
    record = _researcher(tmp_path, StubFactory(state={})).run("AAPL", AS_OF, 0)
    assert record.decision == ""
    assert record.reports == {
        "market_report": "",
        "sentiment_report": "",
        "news_report": "",
        "fundamentals_report": "",
    }


def test_default_factory_builds_the_graph_under_the_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}

    class FakeGraph:
        def __init__(self, *, selected_analysts: list[str], config: dict[str, Any]) -> None:
            seen["locked_at_init"] = ta._GRAPH_LOCK.locked()
            seen["analysts"] = selected_analysts
            seen["config"] = config

        def propagate(self, ticker: str, date_str: str) -> tuple[dict[str, str], str]:
            seen["locked_at_propagate"] = ta._GRAPH_LOCK.locked()
            return {"final_trade_decision": "ok"}, "Hold"

    _install(monkeypatch, "tradingagents.graph.trading_graph", TradingAgentsGraph=FakeGraph)
    record = TradingAgentsResearcher(ResearchConfig(root=tmp_path / "r")).run("MSFT", AS_OF, 0)
    assert record.rating is Rating.HOLD
    assert record.decision == "ok"
    assert seen["locked_at_init"] is True
    assert seen["locked_at_propagate"] is False
    assert not ta._GRAPH_LOCK.locked()
    assert seen["analysts"] == list(ANALYSTS)
    assert seen["config"]["llm_provider"] == "anthropic"
    assert seen["config"]["unrelated"] == "kept"
