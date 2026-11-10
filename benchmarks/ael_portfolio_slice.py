"""Compact AEL portfolio benchmark using the checked-in native price slice.

This is a representative D-full/portfolio contract check, not a full paper
run. Each training case carries ticker, prediction, actual return, tool
evaluation, and numeric outcome fields used by AEL's native rich-memory path.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping

from mematlas.benchmarks.base import BenchmarkAdapter
from mematlas.core.evaluation import BenchmarkCase, Prediction
from mematlas.core.models import MemoryEpisode


_ROOT = Path(__file__).resolve().parents[1]
_PRICE_ROOT = _ROOT / "plan" / "源码" / "AEL" / "code" / "data" / "prices"


def _prices(ticker: str, limit: int = 5) -> list[float]:
    with (_PRICE_ROOT / f"{ticker}.csv").open(newline="", encoding="utf-8") as handle:
        return [float(row["Close"]) for row in list(csv.DictReader(handle))[:limit]]


class AELPortfolioSliceBenchmark(BenchmarkAdapter):
    benchmark_id = "ael_portfolio_slice"

    def __init__(self, *, scope: Mapping[str, str] | None = None) -> None:
        self.scope = dict(scope or {"user_id": "ael-portfolio-slice"})
        self.tickers = ("AAPL", "NVDA", "JNJ")
        self._cases = self._build_cases()

    def _build_cases(self) -> tuple[BenchmarkCase, ...]:
        cases: list[BenchmarkCase] = []
        for bar in range(4):
            for ticker in self.tickers:
                prices = _prices(ticker)
                before, after = prices[bar], prices[bar + 1]
                return_pct = (after - before) / before
                direction = "up" if return_pct >= 0 else "down"
                sector = "healthcare" if ticker == "JNJ" else "technology"
                values = {"ticker": ticker, "sector": sector, "task_type": "price_prediction",
                          "date": f"bar-{bar}", "planner_id": "deterministic-equal-weight",
                          "prediction": {"direction": direction, "magnitude": abs(return_pct)},
                          "actual": {"direction": direction, "return_pct": return_pct},
                          "tool_evaluations": {"momentum": {
                              "output_summary": f"{direction} momentum", "signal": direction, "correct": True}},
                          "episode_index": bar}
                episode = MemoryEpisode(task=f"Predict next return for {ticker}",
                                        reward=max(-1.0, min(1.0, return_pct * 100.0)),
                                        extensions={"ael": values}, scope=self.scope)
                cases.append(BenchmarkCase(
                    case_id=f"train-{ticker.lower()}-{bar}", query=f"{ticker} price prediction",
                    scope=self.scope, episode=episode,
                    metadata={"split": "train", "ticker": ticker, "return_pct": return_pct}))
        cases.append(BenchmarkCase(
            case_id="test-aapl-retrieval", query="AAPL price prediction", scope=self.scope,
            metadata={"split": "test", "ticker": "AAPL", "sector": "technology",
                      "task_type": "price_prediction", "expected_tier": "episodic", "top_k": 4}))
        return tuple(cases)

    def cases(self, split: str = "test") -> Iterable[BenchmarkCase]:
        return (case for case in self._cases
                if split == "all" or case.metadata.get("split") == split)

    def native_score(self, predictions: list[Prediction]) -> Mapping[str, Any]:
        if not predictions:
            return {"ael.episodic_recall": 0.0, "ael.evaluated_cases": 0}
        hits = 0
        rich = 0
        for prediction in predictions:
            expected = prediction.metadata.get("case_metadata", {}).get("expected_tier")
            items = [item for contribution in prediction.applied_memory for item in contribution.items]
            hits += int(any(item.get("memory_type") == expected for item in items))
            rich += int(any("Actual:" in str(item.get("content", "")) and "momentum" in str(item.get("content", ""))
                           for item in items))
        total = len(predictions)
        return {"ael.episodic_recall": hits / total, "ael.rich_tool_ground_truth": rich / total,
                "ael.evaluated_cases": total, "ael.train_episodes": len(tuple(self.cases("train")))}

    def describe(self) -> Mapping[str, Any]:
        return {"benchmark_id": self.benchmark_id, "kind": "native_portfolio_slice",
                "train_cases": len(tuple(self.cases("train"))), "test_cases": len(tuple(self.cases("test"))),
                "source": "checked-in AEL price data; AAPL/NVDA/JNJ four-bar slice",
                "scoring": "episodic retrieval and per-tool ground-truth record",
                "upstream_equivalent": "AEL portfolio D-full"}


__all__ = ["AELPortfolioSliceBenchmark"]
