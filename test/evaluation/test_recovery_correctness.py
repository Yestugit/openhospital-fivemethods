"""Prevent invalid frozen predictions and changed Agents from surviving recovery."""
from dataclasses import replace
from functools import partial
import json

import pytest

from mematlas import ExperimentConfig, ExperimentRunner, MemoryEpisode, MemoryEvent
from mematlas.benchmarks.datasets import InMemoryBenchmark
from mematlas.core.evaluation import BenchmarkCase
from mematlas.evaluation.execution import _agent_identity
from mematlas.evaluation.metrics import Metric, default_metric_registry
from mematlas.test.evaluation.support import FixturePlugin, FixtureMethod, config_for, registry_for


def dataset():
    scope = {"user_id": "alice"}
    return InMemoryBenchmark("recovery", [
        BenchmarkCase("train", "fact", scope=scope, metadata={"split": "train"},
                      history=(MemoryEvent("message", "answer"),)),
        BenchmarkCase("test-0", "question", scope=scope, reference="answer"),
        BenchmarkCase("test-1", "question", scope=scope, reference="answer"),
    ])


def config(tmp_path, **changes):
    return ExperimentConfig.from_dict({"methods": [config_for(tmp_path / "memory")],
        "benchmark": dataset(), "protocol": "experience_train_test",
        "output_dir": str(tmp_path / "results"), **changes})


class Agent:
    def __init__(self, answer="answer", fail_once=False):
        self.config = {"answer": answer}
        self.fail_once, self.calls = fail_once, []

    async def __call__(self, case, memory):
        self.calls.append(case.case_id)
        if case.case_id == "test-1" and self.fail_once:
            self.fail_once = False
            raise RuntimeError("temporary interruption")
        used = await memory.search(case.query, scope=case.scope)
        return MemoryEpisode(output=self.config["answer"], applied_memory=[used],
                             events=[MemoryEvent("message", "answer")])


@pytest.mark.asyncio
async def test_frozen_violation_cannot_be_reused_after_resume(tmp_path):
    class MutatingMethod(FixtureMethod):
        async def search(self, query, **options):
            self.episodes.append({"illegal": "frozen mutation"})
            return await super().search(query, **options)

    class Plugin(FixturePlugin):
        def create_method(self, config, services):
            return MutatingMethod()

    runner = ExperimentRunner(registry=registry_for(Plugin()))
    experiment = config(tmp_path)
    first = (await runner.evaluate(experiment, agent=Agent()))[0]
    assert first.failures and not first.learning_curve
    resumed = (await runner.evaluate(replace(experiment, output_dir=str(tmp_path / "resumed")),
        agent=Agent(), resume_from=tmp_path / "results"))[0]
    assert resumed.failures and not resumed.learning_curve
    assert "Frozen" in str(resumed.failures)


@pytest.mark.asyncio
async def test_changed_agent_settings_reject_resume_before_predicting(tmp_path):
    runner = ExperimentRunner(registry=registry_for())
    experiment = config(tmp_path)
    first = (await runner.evaluate(experiment, agent=Agent(fail_once=True)))[0]
    assert first.failures
    changed = Agent("wrong")
    resumed = (await runner.evaluate(replace(experiment, output_dir=str(tmp_path / "resumed")),
        agent=changed, resume_from=tmp_path / "results"))[0]
    assert resumed.failures and not resumed.predictions
    assert "contract" in str(resumed.failures)
    assert not changed.calls


@pytest.mark.asyncio
async def test_unfinished_checkpoint_retries_predictions_and_keeps_old_operations(tmp_path):
    runner = ExperimentRunner(registry=registry_for())
    experiment, agent = config(tmp_path), Agent(fail_once=True)
    first = (await runner.evaluate(experiment, agent=agent))[0]
    assert first.failures
    agent.calls.clear()
    resumed = (await runner.evaluate(replace(experiment, output_dir=str(tmp_path / "resumed")),
        agent=agent, resume_from=tmp_path / "results"))[0]
    assert not resumed.failures
    assert agent.calls == ["test-0", "test-1"]
    assert resumed.metrics["common.exact_match"] == 1
    assert resumed.resource_metrics["common.add_count"] == 1
    assert sum(row["operation"] == "add" for row in resumed.traces) == 1
    assert runner.results(tmp_path / "resumed")[0].to_dict() == resumed.to_dict()


def test_identity_covers_constructor_defaults_captures_partial_and_bound_method():
    class ConfiguredAgent:
        def __init__(self, answer):
            self.answer, self.calls = answer, []

        async def __call__(self, case, memory):
            return MemoryEpisode(output=self.answer)

    left, right = ConfiguredAgent("yes"), ConfiguredAgent("no")
    assert _agent_identity(left) != _agent_identity(right)
    assert _agent_identity(left.__call__) != _agent_identity(right.__call__)
    identity = _agent_identity(left)
    left.calls.append("runtime log")
    assert _agent_identity(left) == identity
    left.answer = "no"
    assert _agent_identity(left) != identity

    def factory(answer):
        async def captured(case, memory):
            return MemoryEpisode(output=answer)
        return captured
    assert _agent_identity(factory("yes")) != _agent_identity(factory("no"))
    async def defaulted(case, memory, answer="yes"):
        return MemoryEpisode(output=answer)
    initial = _agent_identity(defaulted)
    defaulted.__defaults__ = ("no",)
    assert _agent_identity(defaulted) != initial
    assert _agent_identity(partial(defaulted, answer="yes")) != _agent_identity(partial(defaulted, answer="no"))


class TaskMetric(Metric):
    name = "task_accuracy"

    def compute(self, predictions):
        return sum(row.output == row.reference for row in predictions) / len(predictions)


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["experience_train_test", "experience_learning_curve"])
async def test_custom_metric_survives_final_result_and_reload(tmp_path, protocol):
    metrics = default_metric_registry()
    metrics.register(TaskMetric)
    runner = ExperimentRunner(registry=registry_for(), metric_registry=metrics)
    result = (await runner.evaluate(config(tmp_path, protocol=protocol, metrics=["task_accuracy"]),
                                    agent=Agent()))[0]
    assert not result.failures
    assert result.metrics == {"task_accuracy": 1}
    assert result.learning_curve[-1]["metrics"] == {"task_accuracy": 1}
    assert runner.results(tmp_path / "results")[0].metrics == {"task_accuracy": 1}


def test_longmemeval_instance_checks_judge_and_accepts_supplied_judge(tmp_path):
    from mematlas import MethodServices
    from mematlas.benchmarks.longmemeval import LongMemEvalBenchmark
    path = tmp_path / "longmemeval.json"
    path.write_text(json.dumps([{"question_id": "q", "question": "question", "answer": "answer",
        "question_type": "single-session-user", "haystack_sessions": [],
        "haystack_session_ids": [], "answer_session_ids": []}]), encoding="utf-8")
    experiment = config(tmp_path, protocol="streaming_memory", benchmark=LongMemEvalBenchmark(path))
    runner = ExperimentRunner(registry=registry_for())
    assert not runner.check(experiment, agent=Agent())["ready"]
    assert runner.check(experiment, agent=Agent(),
                        services=MethodServices(objects={"native_judge": lambda prediction: True}))["ready"]


@pytest.mark.asyncio
async def test_changed_injected_reader_cannot_reuse_completed_checkpoint(tmp_path):
    from mematlas import MethodServices
    from mematlas.providers import CallableLLM
    def reader(answer):
        return MethodServices(objects={"answer_llm": CallableLLM(lambda messages, **kwargs: answer)})
    runner, experiment = ExperimentRunner(registry=registry_for()), config(tmp_path)
    first = (await runner.evaluate(experiment, services=reader("answer")))[0]
    assert not first.failures and first.metrics["common.exact_match"] == 1
    resumed = (await runner.evaluate(replace(experiment, output_dir=str(tmp_path / "changed")),
        services=reader("wrong"), resume_from=tmp_path / "results"))[0]
    assert resumed.failures and not resumed.predictions
    assert "contract" in str(resumed.failures)


@pytest.mark.asyncio
async def test_opaque_external_client_allows_fresh_evaluation_but_requires_resume_settings(tmp_path):
    class OpaqueAgent:
        def __init__(self, client):
            self.client = client

        async def __call__(self, case, memory):
            return MemoryEpisode(output="answer")

    runner, experiment = ExperimentRunner(registry=registry_for()), config(tmp_path)
    agent = OpaqueAgent(object())
    first = (await runner.evaluate(experiment, agent=agent))[0]
    assert not first.failures and first.metrics["common.exact_match"] == 1
    assert not first.run_metadata["recovery"]["supported"]
    resumed = (await runner.evaluate(replace(experiment, output_dir=str(tmp_path / "resumed")),
        agent=agent, resume_from=tmp_path / "results"))[0]
    assert resumed.failures and "Cannot verify recovery settings" in str(resumed.failures)
