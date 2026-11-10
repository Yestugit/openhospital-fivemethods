"""Run the compact MemCon ALFWorld/PDDL-shaped benchmark offline."""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from mematlas import ExperimentConfig, ExperimentRunner, MemAtlasConfig, Memory, RuntimeConfig, MemoryUnit
from mematlas.benchmarks.memcon_action_observation import MemConActionObservationBenchmark
from mematlas.core.evaluation import Prediction


async def run():
    benchmark = MemConActionObservationBenchmark()
    with tempfile.TemporaryDirectory(prefix="memcon-action-slice-") as directory:
        config = MemAtlasConfig(method={"method_id": "memcon", "params": {
            "storage_path": directory, "successful_topk": 2, "failed_topk": 1,
            "insight_topk": 2, "top_k": 5}}, runtime=RuntimeConfig(artifact_dir=directory), scope=benchmark.scope)
        predictions = []
        async with Memory(config) as memory:
            for case in benchmark.cases("train"):
                unit = MemoryUnit.from_episode(case.episode)
                receipt = await memory.add(unit)
                await memory.episode_end(receipt.episode_ref)
            for case in benchmark.cases("test"):
                contribution = await memory.search(case.query, top_k=case.metadata["top_k"], scope=case.scope)
                predictions.append(Prediction(case.case_id, None, None, (contribution,),
                                              metadata={"case_metadata": dict(case.metadata)}))
        scores = dict(benchmark.native_score(predictions))
    assert scores["memcon.label_recall"] == 1.0, scores
    assert scores["memcon.source_provenance"] == 1.0, scores
    return scores


async def run_experiment():
    """Exercise the action/observation slice through ExperimentRunner/MemConQA."""
    with tempfile.TemporaryDirectory(prefix="memcon-action-eval-") as directory:
        config = ExperimentConfig.from_dict({
            "methods": [{"method": {"method_id": "memcon", "params": {
                "storage_path": directory, "successful_topk": 2, "failed_topk": 1,
                "insight_topk": 2, "top_k": 5}},
                "scope": {"user_id": "memcon-action-slice"},
                "runtime": {"artifact_dir": directory}}],
            "benchmark": {"kind": "memcon_action_observation"},
            "protocol": "experience_train_test",
            "train_limit": 2,
            "test_limit": 2,
            "output_dir": str(Path(directory) / "evaluation"),
        })
        result = (await ExperimentRunner().evaluate(config))[0]
        assert not result.failures, result.failures
        assert result.metrics["native.memcon.label_recall"] == 1.0, result.metrics
        assert result.metrics["native.memcon.source_provenance"] == 1.0, result.metrics
        return result.metrics


if __name__ == "__main__":
    print(f"MEMCON_ACTION_BENCHMARK {asyncio.run(run())}")
    print(f"MEMCON_ACTION_EXPERIMENT {asyncio.run(run_experiment())}")
