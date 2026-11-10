"""AWM integration must preserve ordinary QA artifacts and legacy recovery."""
from dataclasses import replace

import pytest

from mematlas import ExperimentConfig, ExperimentRunner, MemoryEpisode, MemoryEvent
from mematlas.benchmarks.datasets import InMemoryBenchmark
from mematlas.core.evaluation import BenchmarkCase
from mematlas.core.plugin import MethodServices
from mematlas.evaluation.execution import _EvaluationRun
from mematlas.evaluation.qa import DefaultQA
from mematlas.methods.memrl.qa import MemRLQA
from mematlas.test.awm.reference.evaluation_contract_v3 import _run_training as legacy_training
from mematlas.test.evaluation.support import FixturePlugin, config_for, registry_for


class Reader:
    """No network; keep reader settings independent of runtime call counts."""
    config = {"answer": "answer"}

    def __init__(self):
        self.calls = 0

    async def complete(self, messages, **options):
        self.calls += 1
        return "answer"


def setup_run(tmp_path, qa_type):
    plugin = FixturePlugin()
    plugin.qa_factory = f"{qa_type.__module__}:{qa_type.__qualname__}"
    scope = {"user_id": "alice"}
    benchmark = InMemoryBenchmark("isolation", [
        BenchmarkCase("train", "task", scope=scope, metadata={"split": "train"},
            episode=MemoryEpisode("train", task="task", success=True, output="stored",
                events=[MemoryEvent("message", "answer")])),
        BenchmarkCase("test", "question", scope=scope, reference="answer"),
    ])
    config = ExperimentConfig.from_dict({"methods": [config_for(tmp_path / "memory")],
        "benchmark": benchmark, "protocol": "experience_train_test",
        "output_dir": str(tmp_path / "original")})
    reader = Reader()
    return (ExperimentRunner(registry=registry_for(plugin)), config,
            MethodServices(objects={"answer_llm": reader}), reader, plugin)


@pytest.mark.asyncio
@pytest.mark.parametrize("qa_type", [DefaultQA, MemRLQA])
async def test_ordinary_qa_keeps_prediction_format_and_imported_training(tmp_path, qa_type):
    runner, config, services, reader, plugin = setup_run(tmp_path, qa_type)
    result = (await runner.evaluate(config, services=services))[0]
    assert not result.failures, result.failures
    assert reader.calls == 1  # Supplied training episodes must not call the actor.
    assert plugin.instances[0].episodes[0]["output"] == "stored"
    assert result.predictions[0].output == "answer"
    assert result.predictions[0].applied_memory[0].items
    assert set(result.predictions[0].metadata) == {
        "case_metadata", "episode_id", "response_time_s"}


@pytest.mark.asyncio
@pytest.mark.parametrize("qa_type", [DefaultQA, MemRLQA])
async def test_pre_awm_checkpoint_resumes_without_retraining_or_answering(tmp_path, monkeypatch, qa_type):
    runner, config, services, reader, plugin = setup_run(tmp_path, qa_type)
    # Generate the checkpoint with the frozen original v3 implementation, then
    # resume with the production implementation under review.
    with monkeypatch.context() as original:
        original.setattr(_EvaluationRun, "_run_training", legacy_training)
        first = (await runner.evaluate(config, services=services))[0]
    assert not first.failures, first.failures
    resumed = (await runner.evaluate(replace(config, output_dir=str(tmp_path / "resumed")),
        services=services, resume_from=config.output_dir))[0]
    assert not resumed.failures, resumed.failures
    assert resumed.predictions == first.predictions
    assert reader.calls == 1
    assert not plugin.instances[-1].episodes[1:]
    assert resumed.run_metadata["progress"]["completed_exposures"] == 1
