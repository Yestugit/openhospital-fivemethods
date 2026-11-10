"""Run the compact native ReasoningBank trajectory benchmark.

Usage::

    PYTHONPATH=/home/lzy/project .venv/bin/python \
      benchmarks/check_reasoningbank_trajectory.py

The fixture is deliberately offline and uses a fixed extraction response.  It
checks the same trajectory contract used by WebArena/SWE-Bench without making
paid model requests or requiring either upstream environment.
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from mematlas import ExperimentConfig, ExperimentRunner, MemAtlasConfig, Memory, MethodServices, RuntimeConfig, MemoryUnit
from mematlas.benchmarks.reasoningbank_trajectory import ReasoningBankTrajectoryBenchmark
from mematlas.core.evaluation import Prediction


class FixedExtractionLLM:
    async def complete(self, messages, **kwargs):
        return (
            "# Memory Item 1\n"
            "## Title Verify the workflow state\n"
            "## Description Use this procedure for a related task\n"
            "## Content Inspect the current state, perform the requested action, and verify the result.\n\n"
            "# Memory Item 2\n"
            "## Title Confirm the final result\n"
            "## Description Keep the final tool result as evidence\n"
            "## Content Check the final observation before reporting success."
        )


async def run() -> dict[str, float | int]:
    benchmark = ReasoningBankTrajectoryBenchmark()
    with tempfile.TemporaryDirectory(prefix="reasoningbank-trajectory-") as temporary:
        root = Path(temporary)
        config = MemAtlasConfig(
            method={"method_id": "reasoningbank", "params": {
                "implementation_mode": "offline", "storage_path": str(root / "native"),
                "top_k": 2,
            }},
            runtime=RuntimeConfig(artifact_dir=str(root / "artifacts")),
            scope=benchmark.scope,
        )
        predictions = []
        async with Memory(config, services=MethodServices(llm=FixedExtractionLLM())) as memory:
            for case in benchmark.cases("train"):
                unit = MemoryUnit.from_episode(case.episode)
                receipt = await memory.add(unit)
                await memory.episode_end(receipt.episode_ref)
            for case in benchmark.cases("test"):
                contribution = await memory.search(
                    case.query, top_k=case.metadata.get("top_k"), scope=case.scope)
                predictions.append(Prediction(
                    case.case_id, None, None, (contribution,),
                    metadata={"case_metadata": dict(case.metadata)},
                ))
        scores = dict(benchmark.native_score(predictions))
    assert scores["reasoningbank.trajectory_recall"] == 1.0, scores
    assert scores["reasoningbank.success_label_accuracy"] == 1.0, scores
    assert scores["reasoningbank.source_provenance"] == 1.0, scores
    return scores


async def run_experiment() -> dict[str, float | int]:
    """Exercise the registered benchmark and ReasoningBankQA lifecycle."""
    benchmark = ReasoningBankTrajectoryBenchmark()
    with tempfile.TemporaryDirectory(prefix="reasoningbank-trajectory-eval-") as temporary:
        config = ExperimentConfig.from_dict({
            "methods": [{"method": {"method_id": "reasoningbank", "params": {
                "implementation_mode": "offline", "storage_path": str(Path(temporary) / "native"),
                "top_k": 2}},
                "scope": benchmark.scope,
                "runtime": {"artifact_dir": str(Path(temporary) / "artifacts")}}],
            "benchmark": {"kind": "reasoningbank_trajectory"},
            "protocol": "experience_train_test",
            "train_limit": 3,
            "test_limit": 3,
            "output_dir": str(Path(temporary) / "evaluation"),
        })
        result = (await ExperimentRunner().evaluate(
            config, services=MethodServices(llm=FixedExtractionLLM())))[0]
        assert not result.failures, result.failures
        for name in ("trajectory_recall", "success_label_accuracy", "source_provenance"):
            assert result.metrics[f"native.reasoningbank.{name}"] == 1.0, result.metrics
        return dict(result.metrics)


if __name__ == "__main__":
    print(asyncio.run(run()))
    print(f"REASONINGBANK_EXPERIMENT {asyncio.run(run_experiment())}")
