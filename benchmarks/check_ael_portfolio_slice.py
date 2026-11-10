"""Run the compact AEL portfolio benchmark without external services."""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from mematlas import ExperimentConfig, ExperimentRunner, MemAtlasConfig, Memory, RuntimeConfig, MemoryUnit
from mematlas.benchmarks.ael_portfolio_slice import AELPortfolioSliceBenchmark
from mematlas.core.evaluation import Prediction


async def run():
    benchmark = AELPortfolioSliceBenchmark()
    with tempfile.TemporaryDirectory(prefix="ael-portfolio-slice-") as directory:
        config = MemAtlasConfig(method={"method_id": "ael", "params": {
            "runner_mode": "episode", "memory_policy_names": ["full_detailed"],
            "storage_path": directory, "retrieval_top_k": 4}},
            runtime=RuntimeConfig(artifact_dir=directory), scope=benchmark.scope)
        predictions = []
        async with Memory(config) as memory:
            for case in benchmark.cases("train"):
                unit = MemoryUnit.from_episode(case.episode)
                receipt = await memory.add(unit)
                await memory.episode_end(receipt.episode_ref)
            for case in benchmark.cases("test"):
                contribution = await memory.search(case.query, top_k=case.metadata["top_k"], scope=case.scope,
                                                     extensions={"ael": {"ticker": case.metadata["ticker"],
                                                                          "sector": "healthcare" if case.metadata["ticker"] == "JNJ" else "technology",
                                                                          "task_type": "price_prediction"}})
                predictions.append(Prediction(case.case_id, None, None, (contribution,),
                                              metadata={"case_metadata": dict(case.metadata)}))
        scores = dict(benchmark.native_score(predictions))
    assert scores["ael.episodic_recall"] == 1.0, scores
    assert scores["ael.rich_tool_ground_truth"] == 1.0, scores
    return scores


async def run_experiment():
    """Exercise the same slice through ExperimentRunner and AELQA."""
    with tempfile.TemporaryDirectory(prefix="ael-portfolio-eval-") as directory:
        config = ExperimentConfig.from_dict({
            "methods": [{"method": {"method_id": "ael", "params": {
                "runner_mode": "episode", "memory_policy_names": ["full_detailed"],
                "storage_path": directory, "retrieval_top_k": 4}},
                "scope": {"user_id": "ael-portfolio-slice"},
                "runtime": {"artifact_dir": directory}}],
            "benchmark": {"kind": "ael_portfolio_slice"},
            "protocol": "experience_train_test",
            "train_limit": 12,
            "test_limit": 1,
            "output_dir": str(Path(directory) / "evaluation"),
        })
        result = (await ExperimentRunner().evaluate(config))[0]
        assert not result.failures, result.failures
        assert result.metrics["native.ael.episodic_recall"] == 1.0, result.metrics
        assert result.metrics["native.ael.rich_tool_ground_truth"] == 1.0, result.metrics
        return result.metrics


if __name__ == "__main__":
    print(f"AEL_PORTFOLIO_BENCHMARK {asyncio.run(run())}")
    print(f"AEL_PORTFOLIO_EXPERIMENT {asyncio.run(run_experiment())}")
