"""Regression checks for missing answers, abstention and native judge labels."""
from dataclasses import replace
import re

import pytest

from mematlas import ExperimentConfig, ExperimentRunner, MemoryEpisode, MethodServices
from mematlas.benchmarks.datasets import InMemoryBenchmark
from mematlas.benchmarks.locomo import LoCoMoBenchmark
from mematlas.benchmarks.longmemeval import LongMemEvalBenchmark, LongMemEvalLLMJudge, _judge_label
from mematlas.core.evaluation import BenchmarkCase, Prediction
from mematlas.core.models import MemoryContribution
from mematlas.core.telemetry import TraceCollector, UsageLedger
from mematlas.evaluation.metrics import Evaluator, ExactMatchMetric, TokenF1Metric
from mematlas.evaluation.qa import DefaultQA
from mematlas.providers import CallableLLM
from mematlas.test.evaluation.support import config_for, registry_for


@pytest.mark.parametrize("metric", [ExactMatchMetric(), TokenF1Metric()])
def test_answer_metrics_require_references_and_count_missing_answers_as_wrong(metric):
    predictions = [
        Prediction("correct", "answer", "answer"),
        Prediction("missing", None, "None"),
        Prediction("failed", "answer", "answer", metadata={"evaluation_error": "reader failed"}),
        Prediction("unlabeled", "anything", None),
        Prediction("empty", None, None),
    ]
    assert metric.compute(predictions) == pytest.approx(1 / 3)
    assert metric.compute(predictions[3:]) is None
    assert metric.compute([]) is None


def test_coverage_separates_missing_outputs_from_exceptions():
    resources = {}
    predictions = [Prediction("complete", "answer", None), Prediction("missing", None, None),
                   Prediction("failed", None, "answer", metadata={"evaluation_error": "reader failed"})]
    Evaluator._add_coverage_resources(resources, predictions, 4)
    assert resources["common.expected_case_count"] == 4
    assert resources["common.completed_case_count"] == 1
    assert resources["common.failed_case_count"] == 1
    assert resources["common.missing_output_case_count"] == 1
    assert resources["common.case_coverage"] == 0.25


@pytest.mark.asyncio
async def test_empty_agent_cannot_get_full_marks_without_a_reference(tmp_path):
    benchmark = InMemoryBenchmark("unlabeled", [BenchmarkCase("empty", "question")])
    config = ExperimentConfig.from_dict({"methods": [config_for(tmp_path / "memory")],
        "benchmark": benchmark, "output_dir": str(tmp_path / "results")})

    async def agent(case, memory):
        return MemoryEpisode()

    result = (await ExperimentRunner(registry=registry_for()).evaluate(config, agent=agent))[0]
    assert result.metrics["common.exact_match"] is None
    assert result.metrics["common.token_f1"] is None
    assert result.resource_metrics["common.completed_case_count"] == 0
    assert result.resource_metrics["common.missing_output_case_count"] == 1
    assert result.resource_metrics["common.case_coverage"] == 0


@pytest.mark.asyncio
async def test_default_reader_abstention_matches_locomo_native_scoring(tmp_path):
    def obey_prompt(messages, **kwargs):
        return re.search(r'reply exactly "([^"]+)"', messages[-1]["content"]).group(1)

    case = BenchmarkCase("abstention", "What is the unknown detail?", reference="unknown",
                         metadata={"category": 5})
    class EmptyMemory:
        async def search(self, query, **options):
            return MemoryContribution()

    qa = DefaultQA(None, MethodServices(objects={"answer_llm": CallableLLM(obey_prompt)}))
    episode = await qa(case, EmptyMemory())
    prediction = Prediction(case.case_id, episode.output, case.reference,
                            tuple(episode.applied_memory), metadata={"case_metadata": case.metadata})
    path = tmp_path / "locomo.json"
    path.write_text("[]", encoding="utf-8")
    assert LoCoMoBenchmark(path).native_score([prediction])["qa_f1.category_5"] == 1


@pytest.mark.parametrize("value,expected", [
    (True, True), (False, False), ("yes", True), (" No. ", False),
    ({"correct": "no"}, False), ({"label": "false"}, False),
    ({"correct": "true"}, True), ({"label": 0}, False), ({"correct": 1}, True),
])
def test_judge_labels_are_explicit(value, expected):
    assert _judge_label(value) is expected


@pytest.mark.parametrize("value", [None, {}, {"correct": "maybe"}, "yesterday", "yes or no", 2])
def test_unrecognized_judge_labels_raise_clear_errors(value):
    with pytest.raises(ValueError, match="Unrecognized LongMemEval judge label"):
        _judge_label(value)


@pytest.mark.asyncio
async def test_native_judge_mapping_no_scores_zero_and_invalid_label_is_a_scoring_failure(tmp_path):
    path = tmp_path / "longmemeval.json"
    path.write_text("[]", encoding="utf-8")
    predictions = [Prediction("test", "incorrect", "correct",
                               metadata={"case_metadata": {"question_type": "single-session-user"}})]
    benchmark = LongMemEvalBenchmark(path, judge=lambda prediction: {"correct": "no"})
    values = await benchmark.native_score_async(predictions)
    assert values["judge_accuracy"] == 0
    assert values["judge_task_accuracy"] == 0

    benchmark.judge = lambda prediction: {"correct": "unrecognized"}
    metrics, _ = await Evaluator().evaluate(predictions, benchmark=benchmark,
        traces=TraceCollector(), artifact_dir=tmp_path)
    assert "ValueError: Unrecognized LongMemEval judge label" in metrics["native.scoring_error"]
    assert "native.judge_accuracy" not in metrics


@pytest.mark.asyncio
async def test_native_llm_judge_accounts_for_custom_provider_calls(tmp_path):
    path = tmp_path / "longmemeval.json"
    path.write_text("[]", encoding="utf-8")
    benchmark = LongMemEvalBenchmark(path,
        judge=LongMemEvalLLMJudge(CallableLLM(lambda messages, **kwargs: "no")))
    prediction = Prediction("test", "incorrect", "correct", metadata={"case_metadata": {
        "question_type": "single-session-user", "question": "question"}})
    usage = UsageLedger()
    metrics, resources = await Evaluator().evaluate([prediction], benchmark=benchmark,
        traces=TraceCollector(), artifact_dir=tmp_path, usage=usage)
    assert metrics["native.judge_accuracy"] == 0
    assert resources["common.judge_llm_calls"] == 1
    assert len(usage.all()) == 1
    assert usage.all()[0].phase == "judge"
    assert not usage.all()[0].usage_known
