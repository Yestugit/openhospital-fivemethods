"""Public evaluation contracts: one config, default QA, scheduling and recovery."""
import inspect
import json
from dataclasses import replace

import pytest

from mematlas import ExperimentConfig, ExperimentRunner, MemoryEpisode, MemoryEvent, MethodServices
from mematlas.benchmarks.datasets import InMemoryBenchmark
from mematlas.core.evaluation import BenchmarkCase, EvaluationResult, Prediction
from mematlas.providers import CallableLLM
from mematlas.test.evaluation.support import FixturePlugin, config_for, registry_for


def data():
    scope = {"user_id": "alice"}
    return InMemoryBenchmark("unified", [
        *[BenchmarkCase(f"train-{i}", f"task {i}", scope=scope,
            history=(MemoryEvent("message", f"fact {i}", event_id=f"event-{i}"),),
            episode=MemoryEpisode(f"episode-{i}", task=f"task {i}",
                events=[MemoryEvent("message", f"fact {i}", event_id=f"event-{i}")],
                success=bool(i % 2), scope=scope),
            metadata={"split": "train"}) for i in range(4)],
        *[BenchmarkCase(f"test-{i}", "question", scope=scope, reference="answer") for i in range(3)],
    ])


def experiment(tmp_path, **overrides):
    return ExperimentConfig.from_dict({
        "methods": [config_for(tmp_path / "memory")], "benchmark": data(),
        "protocol": "experience_learning_curve", "checkpoints": (0, 1, 3),
        "train_limit": 3, "test_limit": 2, "output_dir": str(tmp_path / "results"),
        **overrides,
    })


def reader():
    return MethodServices(objects={"answer_llm": CallableLLM(lambda messages, **kwargs: "answer")})


@pytest.mark.asyncio
async def test_default_qa_and_limits_use_one_pipeline(tmp_path):
    plugin = FixturePlugin(required_feedback=True)
    runner = ExperimentRunner(registry=registry_for(plugin))
    result = (await runner.evaluate(experiment(tmp_path), services=reader()))[0]
    assert not result.failures
    assert result.metrics["common.exact_match"] == 1
    assert [row["checkpoint"] for row in result.learning_curve] == [0, 1, 3]
    assert all([p["case_id"] for p in row["predictions"]] == ["test-0", "test-1"]
               for row in result.learning_curve)
    assert len(plugin.instances[0].episodes) == 3
    assert [row["success"] for row in plugin.instances[0].episodes] == [False, True, False]
    assert all(row["frozen_verified"] for row in result.learning_curve)
    assert result.run_metadata["progress"]["completed_exposures"] == 3
    assert result.run_metadata["progress"]["learned_exposures"] == 3
    assert "evolve" not in {row["operation"] for row in result.traces}
    assert runner.results(tmp_path / "results")[0].to_dict() == result.to_dict()
    assert not (tmp_path / "results" / "result.json").exists()


@pytest.mark.asyncio
async def test_single_agent_is_reused_for_training_and_frozen_tests(tmp_path):
    calls = []
    plugin = FixturePlugin()
    async def agent(case, memory):
        calls.append((case.case_id, memory.is_read_only))
        contribution = await memory.search(case.query, scope=case.scope)
        return MemoryEpisode(task=case.query, output="answer", applied_memory=[contribution],
            events=[MemoryEvent("message", f"actual {case.case_id}", role="assistant")])
    runner = ExperimentRunner(registry=registry_for(plugin))
    result = (await runner.evaluate(experiment(tmp_path), agent=agent))[0]
    assert not result.failures
    assert [identity for identity, frozen in calls if not frozen] == ["train-0", "train-1", "train-2"]
    assert [identity for identity, frozen in calls if frozen] == ["test-0", "test-1"] * 3
    assert len(plugin.instances[0].episodes) == 3
    assert all(p.applied_memory for p in result.predictions)


@pytest.mark.asyncio
async def test_train_test_only_measures_after_training(tmp_path):
    runner = ExperimentRunner(registry=registry_for())
    config = experiment(tmp_path, protocol="experience_train_test", checkpoints=())
    result = (await runner.evaluate(config, services=reader()))[0]
    assert not result.failures
    assert [row["checkpoint"] for row in result.learning_curve] == [3]


@pytest.mark.asyncio
async def test_multiple_methods_are_isolated_and_results_share_one_format(tmp_path):
    plugin = FixturePlugin()
    config = experiment(tmp_path, methods=[config_for(tmp_path / "first"), config_for(tmp_path / "second")])
    runner = ExperimentRunner(registry=registry_for(plugin))
    results = await runner.evaluate(config, services=reader())
    assert len(results) == len(plugin.instances) == 2
    assert all(len(method.episodes) == 3 for method in plugin.instances)
    assert all(not result.failures for result in results)
    saved = runner.results(tmp_path / "results")
    assert [r.to_dict() for r in saved] == [r.to_dict() for r in results]
    assert all(isinstance(r, EvaluationResult) for r in saved)


@pytest.mark.parametrize("change,error", [
    ({"checkpoints": (0, 5)}, "checkpoint"),
    ({"test_limit": 100, "split": "missing"}, "No test"),
    ({"train_split": "missing"}, "No training"),
])
def test_invalid_plans_are_rejected_before_memory_creation(tmp_path, change, error):
    plugin = FixturePlugin()
    runner = ExperimentRunner(registry=registry_for(plugin))
    report = runner.check(experiment(tmp_path, **change), services=reader())
    assert not report["ready"] and error in str(report)
    assert not plugin.instances


def test_missing_model_or_feedback_is_reported_by_same_check(tmp_path):
    plugin = FixturePlugin(required_feedback=True)
    runner = ExperimentRunner(registry=registry_for(plugin))
    report = runner.check(experiment(tmp_path))
    assert not report["ready"] and "answer_model" in str(report)
    dataset = data()
    first = dataset._cases[0]
    dataset._cases = (replace(first, episode=replace(first.episode, success=None)), *dataset._cases[1:])
    report = runner.check(experiment(tmp_path, benchmark=dataset), services=reader())
    assert not report["ready"] and "success" in str(report)
    assert not plugin.instances


@pytest.mark.asyncio
async def test_evaluate_checks_automatically_without_allocating_memory(tmp_path):
    plugin = FixturePlugin()
    runner = ExperimentRunner(registry=registry_for(plugin))
    with pytest.raises(ValueError, match="not ready"):
        await runner.evaluate(experiment(tmp_path))
    assert not plugin.instances and not (tmp_path / "results").exists()


@pytest.mark.asyncio
async def test_resume_skips_completed_training_and_checkpoint_cases(tmp_path):
    plugin = FixturePlugin()
    failed = {"enabled": True}
    calls = []
    async def agent(case, memory):
        calls.append((case.case_id, memory.is_read_only))
        if case.case_id == "train-1" and failed["enabled"]:
            raise RuntimeError("interrupted training")
        used = await memory.search(case.query, scope=case.scope)
        return MemoryEpisode(task=case.query, output="answer", applied_memory=[used],
                             events=[MemoryEvent("message", case.case_id)])
    agent.config = {"answer": "answer"}  # Failure injection and call logs are runtime state.
    runner = ExperimentRunner(registry=registry_for(plugin))
    config = experiment(tmp_path)
    first = (await runner.evaluate(config, agent=agent))[0]
    assert first.failures
    assert first.run_metadata["progress"]["completed_exposures"] == 1
    failed["enabled"] = False
    calls.clear()
    resumed = (await runner.evaluate(replace(config, output_dir=str(tmp_path / "resumed")),
                                    agent=agent, resume_from=tmp_path / "results"))[0]
    assert not resumed.failures, resumed.failures
    assert ("train-0", False) not in calls
    assert resumed.run_metadata["progress"]["completed_exposures"] == 3
    assert [row["checkpoint"] for row in resumed.learning_curve] == [0, 1, 3]
    assert runner.results(tmp_path / "resumed")[0].to_dict() == resumed.to_dict()


def test_public_interface_has_no_old_execution_branches():
    import mematlas.evaluation as evaluation
    assert not any(hasattr(evaluation, name) for name in (
        "ExperimentSpec", "ExperiencePlan", "MemoryExperience", "ExperienceStreamPlan"))
    signature = inspect.signature(ExperimentRunner.evaluate)
    assert not {"answerer", "protocol", "benchmark", "memory", "train_batch", "test_batch", "output_dir"} & set(signature.parameters)


def test_config_file_resolves_inputs_relative_to_itself(tmp_path):
    (tmp_path / "method.json").write_text(json.dumps(config_for(tmp_path / "memory")), encoding="utf-8")
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps({"methods": ["method.json"], "benchmark": {"kind": "custom", "path": "cases.json"}}),
                    encoding="utf-8")
    config = ExperimentConfig.load(path)
    assert config.methods[0].method.method_id == "fixture"
    assert config.benchmark["path"] == str((tmp_path / "cases.json").resolve())
    assert ExperimentConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()


def test_final_results_omit_fees_and_preserve_usage_and_sources(tmp_path):
    from mematlas.core.telemetry import UsageLedger, UsageRecord
    from mematlas.evaluation.reporting import ReportWriter

    ledger = UsageLedger()
    ledger.add(UsageRecord("test", "model", prompt_tokens=5, completion_tokens=2,
                           cost=0.007, usage_source="reported"))
    raw_usage = ledger.to_dicts()
    prediction = Prediction("case", {"cost": "task content"}, "reference",
                            metadata={"source_event_ids": ["event-1"]})
    resources = {**ledger.resource_metrics(), "common.run_wall_time_ms": 100,
                 "common.storage_bytes": 64, "common.usage_source": "reported"}
    point = {"checkpoint": 1, "combined_cost": 0.007, "cumulative_acquisition_cost": 0.007,
             "cost_currency": "USD", "cost_coverage": "complete",
             "combined_total_tokens": 7, "window_wall_time_ms": 100,
             "window_usage": ledger.summary(), "resource_metrics": resources,
             "predictions": [prediction.to_dict()]}
    result = EvaluationResult("fixture", "dataset", (prediction,), {"common.exact_match": 1},
        resources, run_metadata={"cost_summary": {"total": ledger.summary()},
            "method_source": {"git_revision": "revision"}},
        traces=({"usage_source": "reported", "source_event_ids": ["event-1"]},),
        learning_curve=(point,))
    ReportWriter().write([result], tmp_path)
    saved = ExperimentRunner.results(tmp_path)[0]

    assert saved.to_dict() == result.to_dict()
    for values in (saved.resource_metrics, saved.learning_curve[0]["resource_metrics"]):
        assert not any("cost" in key or "currency" in key for key in values
                       if not key.endswith(("costed_calls", "uncosted_calls")))
        assert values["common.total_tokens"] == 7
        assert values["common.run_wall_time_ms"] == 100
        assert values["common.storage_bytes"] == 64
        assert values["common.usage_source"] == "reported"
    assert "cost_summary" not in saved.run_metadata
    assert saved.run_metadata["method_source"] == {"git_revision": "revision"}
    curve = saved.learning_curve[0]
    assert "combined_cost" not in curve and "cumulative_acquisition_cost" not in curve
    assert "cost_currency" not in curve and "cost" not in curve["window_usage"]
    assert curve["cost_coverage"] == "complete" and curve["window_usage"]["total_tokens"] == 7
    assert curve["predictions"][0]["output"] == {"cost": "task content"}
    assert saved.predictions[0].metadata["source_event_ids"] == ["event-1"]
    assert saved.traces[0]["usage_source"] == "reported"
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "| common.cost |" not in report and "| common.total_evaluation_cost |" not in report
    assert ledger.to_dicts() == raw_usage and ledger.summary()["cost"] == 0.007

