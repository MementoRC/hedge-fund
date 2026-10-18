"""TradingAgents-backed Researcher: one isolated, freshly configured graph run per call."""

from __future__ import annotations

import importlib
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol, cast

from hedge_fund.domain.research import Rating, RunRecord

# TradingAgentsGraph.__init__ merges into a process-global config dict without a lock.
_GRAPH_LOCK = threading.Lock()
_REPORT_KEYS = ("market_report", "sentiment_report", "news_report", "fundamentals_report")
_DECISION_KEY = "final_trade_decision"


class Graph(Protocol):
    """Anything with TradingAgentsGraph's ``propagate`` shape."""

    def propagate(self, ticker: str, date_str: str) -> tuple[Mapping[str, Any], str]: ...


GraphFactory = Callable[[dict[str, Any], Sequence[str]], Graph]


@dataclass(frozen=True)
class ResearchConfig:
    """Where per-run scratch dirs live, and which models and analysts TradingAgents uses."""

    root: Path
    provider: str = "anthropic"
    deep_model: str = "claude-opus-5-5"
    quick_model: str = "claude-sonnet-5"
    debate_rounds: int = 1
    risk_rounds: int = 1
    analysts: tuple[str, ...] = ("market", "social", "news", "fundamentals")


class TradingAgentsResearcher:
    """Researcher running the forked TradingAgents, isolated per run.

    ``graph_factory`` builds the graph from (config, analysts); None means the real
    TradingAgentsGraph, lazily imported so this module loads without tradingagents.
    """

    def __init__(self, config: ResearchConfig, graph_factory: GraphFactory | None = None) -> None:
        self._config = config
        self._factory: GraphFactory = graph_factory or _tradingagents_graph

    def run(self, ticker: str, as_of: date, run_index: int) -> RunRecord:
        """One full TradingAgents pass; raises on any failure (the service records it).

        Each run gets a fresh scratch dir under ``root`` (results, cache and an empty
        memory log, so no reflection calls and no cross-run state), removed on exit.
        An unknown signal string raises ValueError.
        """
        self._config.root.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(dir=self._config.root))
        try:
            graph = self._factory(self._graph_config(work), self._config.analysts)
            started = time.perf_counter()
            state, signal = graph.propagate(ticker, as_of.isoformat())
            seconds = time.perf_counter() - started
            return RunRecord(
                ticker=ticker,
                as_of=as_of,
                run_index=run_index,
                rating=Rating(signal),
                decision=str(state.get(_DECISION_KEY) or ""),
                reports={key: str(state.get(key) or "") for key in _REPORT_KEYS},
                seconds=seconds,
                error=None,
            )
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _graph_config(self, work: Path) -> dict[str, Any]:
        """Fresh TradingAgents config; our keys are set after the env-folding base call."""
        config = _base_config()
        config.update(
            {
                "llm_provider": self._config.provider,
                "deep_think_llm": self._config.deep_model,
                "quick_think_llm": self._config.quick_model,
                "max_debate_rounds": self._config.debate_rounds,
                "max_risk_discuss_rounds": self._config.risk_rounds,
                "results_dir": str(work / "results"),
                "data_cache_dir": str(work / "cache"),
                "memory_log_path": str(work / "memory" / "trading_memory.md"),
            }
        )
        return config


def _base_config() -> dict[str, Any]:
    module = importlib.import_module("tradingagents.default_config")
    return dict(module.build_default_config())


def _tradingagents_graph(config: dict[str, Any], analysts: Sequence[str]) -> Graph:
    """Build the real graph; construction is serialized, ``propagate`` runs outside the lock."""
    module = importlib.import_module("tradingagents.graph.trading_graph")
    with _GRAPH_LOCK:
        return cast(
            Graph, module.TradingAgentsGraph(selected_analysts=list(analysts), config=config)
        )
