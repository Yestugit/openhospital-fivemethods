"""Academic measurement boundaries, units, completeness and recovery."""
import math
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from mematlas import ExperimentConfig, ExperimentRunner, MemoryEpisode, MemoryEvent, MethodServices
from mematlas.benchmarks.datasets import InMemoryBenchmark
from mematlas.benchmarks.longmemeval import _retrieval_metrics
from mematlas.core.evaluation import BenchmarkCase, EvaluationResult, Prediction
from mematlas.core.files import storage_bytes
from mematlas.core.models import OperationTrace
from mematlas.core.telemetry import ProcessResourceMonitor, TraceCollector, UsageLedger, UsageRecord, usage_context, usage_scope
from mematlas.evaluation.metrics.common import Evaluator
from mematlas.evaluation.reporting import ReportWriter
from mematlas.providers import CallableLLM
from mematlas.test.evaluation.support import FixtureMethod, FixturePlugin, config_for, registry_for


def test_stage_times_exclude_recovery_and_scoring_from_memory_updates():
    traces = TraceCollector()
    for operation, stage, seconds in (
        ("setup", "preparation", 1), ("add", "construction", 2),
        ("flush", "construction", 3), ("add", "training", 4),
        ("add", "training", 5), ("episode_end", "training", 6),
        ("flush", "training", 7), ("save", "training", 100),
        ("load", "recovery", 100), ("add", "evaluation", 100),
        ("search", "evaluation", 0.05), ("judge_scoring", "evaluation", 8),
    ):
        traces.add(OperationTrace("fixture", operation, "now", seconds * 1000, True,
                                  metadata={"stage": stage}))
    traces.add(OperationTrace("fixture", "search", "now", 100, False))
    values = Evaluator._operation_resources(traces)
    assert values["common.memory_construction_time_s"] == 6
    assert values["common.memory_learning_time_s"] == 22
    assert values["common.judging_time_s"] == 8
    assert values["common.retrieval_mean_ms"] == 50
    assert values["common.retrieval_count"] == values["common.retrieval_failures"] == 1


def test_request_kinds_roles_and_unknown_usage_remain_distinct():
    ledger = UsageLedger()
    for phase, kind, inputs, outputs, known in (
        ("memory", "llm", 10, 2, True), ("memory", "embedding", 7, 0, True),
        ("memory", "llm", 0, 0, False), ("agent", "llm", 20, 3, True),
        ("doctor", "llm", 30, 4, True), ("judge", "llm", 40, 5, True),
    ):
        ledger.add(UsageRecord("fixture", "model", prompt_tokens=inputs, completion_tokens=outputs,
            phase=phase, resource_kind=kind, usage_known=known,
            status="failed" if not known else "success"))
    values = ledger.resource_metrics()
    assert values["common.llm_calls"] == 4
    assert values["common.embedding_calls"] == 1
    assert values["common.judge_llm_calls"] == 1
    assert values["common.agent_prompt_tokens"] == 50
    assert values["common.memory_llm_calls"] == 2
    assert values["common.memory_embedding_calls"] == 1
    assert values["common.memory_embedding_total_tokens"] == 7
    assert values["common.total_tokens"] is None
    assert values["common.prompt_tokens"] is None
    assert values["common.completion_tokens"] is None
    assert values["common.known_prompt_tokens"] == 67
    assert values["common.memory_token_coverage"] == "partial"
    assert ledger.summary()["prompt_tokens"] == 107
    assert ledger.summary(resource_kind="embedding")["tracked_calls"] == 1


def test_cache_reporting_keeps_missing_distinct_from_zero_and_restores(tmp_path):
    ledger = UsageLedger()
    ledger.bind_journal(tmp_path / "usage.jsonl")
    for hit, created in ((0, 1500), (1500, 0)):
        ledger.add(UsageRecord("fixture", "model", prompt_tokens=2000, metadata={
            "prompt_tokens_details": {"cached_tokens": hit, "cache_creation_input_tokens": created}}))
    ledger.add(UsageRecord("local", "embedding", prompt_tokens=9000, resource_kind="embedding"))
    summary = ledger.summary()
    assert summary["prompt_tokens"] == 13000  # Cached tokens are a subset, never subtracted or added.
    assert summary["cached_prompt_tokens"] == 1500 and summary["known_cache_creation_prompt_tokens"] == 1500
    assert summary["prompt_cache_hit_rate"] == pytest.approx(1500 / 4000)
    assert summary["cache_token_coverage"] == "complete"
    ledger.add(UsageRecord("legacy", "model", prompt_tokens=2000))
    assert ledger.summary()["cached_prompt_tokens"] is None
    assert ledger.summary()["cache_token_coverage"] == "partial"
    assert ledger.summary()["prompt_cache_hit_rate"] == pytest.approx(1500 / 4000)
    restored = UsageLedger()
    restored.import_journal(tmp_path / "usage.jsonl")
    assert restored.summary() == ledger.summary()
    unknown = UsageLedger(_records=[UsageRecord("legacy", "model", prompt_tokens=2000)])
    assert unknown.summary()["prompt_cache_hit_rate"] is None
    assert unknown.summary()["cache_token_coverage"] == "unknown"


@pytest.mark.asyncio
async def test_provider_preserves_reported_cache_details():
    from mematlas.providers.requests import metered_request
    ledger = UsageLedger()
    details = {"cached_tokens": 1200, "cache_write_tokens": 0}
    async def request():
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=1600, completion_tokens=1,
            prompt_tokens_details=SimpleNamespace(model_dump=lambda **kwargs: details)))
    await metered_request(request, ledger=ledger, provider="fixture", model="model", kind="llm")
    assert ledger.all()[0].metadata["prompt_tokens_details"] == details
    assert ledger.summary()["prompt_cache_hit_rate"] == 0.75


def test_response_timing_marks_missing_measurements():
    predictions = [Prediction(str(i), "answer", "answer", metadata={"response_time_s": value})
                   for i, value in enumerate((2, 4, None, float("inf"), float("nan"), -1, True))]
    predictions.append(Prediction("failure", None, "answer",
                                  metadata={"evaluation_error": "failed", "response_time_s": 50}))
    values = Evaluator._response_resources(predictions)
    assert values["common.response_mean_s"] == 3
    assert values["common.response_count"] == 2
    assert values["common.response_timing_coverage"] == pytest.approx(2 / 7)
    assert Evaluator._response_resources([])["common.response_mean_s"] is None


def test_live_storage_deduplicates_paths_and_excludes_recovery_copies(tmp_path):
    live = tmp_path / "data"
    live.mkdir()
    (live / "index.db").write_bytes(b"x" * 1024 ** 2)
    (live / "vectors.bin").write_bytes(b"y" * 128)
    (live / "usage.jsonl").write_text("log", encoding="utf-8")
    (live / "native.log").write_text("log", encoding="utf-8")
    for name in ("snapshots", "state-snapshots", "adapter_snapshots", "checkpoints", "logs"):
        folder = live / name
        folder.mkdir()
        (folder / "index.db").write_bytes(b"x" * 4096)
    size = storage_bytes((live, live / "index.db", live / "snapshots", live / "adapter_snapshots"))
    assert size == 1024 ** 2 + 128
    assert Evaluator._storage_resources(size)["common.memory_storage_mib"] == size / 1024 ** 2
    assert Evaluator._storage_resources(None)["common.memory_storage_mib"] is None


@pytest.mark.asyncio
async def test_peak_rss_uses_mib_and_preserves_unavailable_state(monkeypatch):
    monitor = ProcessResourceMonitor()
    samples = iter((2 * 1024 ** 2, 3 * 1024 ** 2, 1024 ** 2))
    monkeypatch.setattr(monitor, "_total_rss", lambda: next(samples))
    monitor._sample()
    monitor._sample()
    resources = await monitor.stop()
    assert resources["process_peak_rss_mib"] == 3
    assert resources["process_memory_sample_count"] == 3
    assert (await ProcessResourceMonitor().stop())["process_peak_rss_mib"] is None


@pytest.mark.parametrize("rank,expected", [(1, 1), (2, 1 / math.log2(3)), (3, 0.5)])
def test_ndcg_has_standard_logarithmic_discounts(rank, expected):
    ranked = ["irrelevant-1", "irrelevant-2"]
    ranked.insert(rank - 1, "relevant")
    recall_any, recall_all, ndcg = _retrieval_metrics(ranked, {"relevant"}, 3)
    assert recall_any == recall_all == 1
    assert ndcg == pytest.approx(expected)
    assert _retrieval_metrics(["a", "b"], {"a", "b"}, 3)[2] == 1


class Clock:
    value = 0

    def now(self):
        return self.value

    def advance(self, value):
        self.value += value


def timed_run(tmp_path, monkeypatch):
    clock = Clock()
    monkeypatch.setattr("mematlas.evaluation.execution.perf_counter", clock.now)
    monkeypatch.setattr("mematlas.evaluation.metrics.common.perf_counter", clock.now)

    class TimedMethod(FixtureMethod):
        async def _learn(self, episode, **options):
            clock.advance(100)
            await super()._learn(episode, **options)

        async def search(self, query, **options):
            clock.advance(2)
            return await super().search(query, **options)

    class TimedPlugin(FixturePlugin):
        def create_method(self, config, services):
            method = TimedMethod()
            self.instances.append(method)
            return method

    class TimedBenchmark(InMemoryBenchmark):
        async def native_score_async(self, predictions):
            clock.advance(40)
            return {"score": 1}

    def answer(messages, **options):
        clock.advance(3)
        return "answer"
    answer.config = {}  # The clock measures execution; it does not configure answers.

    scope = {"user_id": "alice"}
    dataset = TimedBenchmark("timed", [
        BenchmarkCase("train", "task", scope=scope, metadata={"split": "train"},
                      episode=MemoryEpisode(task="task", events=[MemoryEvent("message", "fact")])),
        BenchmarkCase("test", "question", scope=scope, reference="answer",
                      history=(MemoryEvent("message", "fact"),)),
    ])
    runner = ExperimentRunner(registry=registry_for(TimedPlugin()))
    services = MethodServices(objects={"answer_llm": CallableLLM(answer)})
    config = ExperimentConfig.from_dict({"methods": [config_for(tmp_path / "memory")],
        "benchmark": dataset, "output_dir": str(tmp_path / "results")})
    return runner, config, services, clock


@pytest.mark.asyncio
async def test_qa_response_excludes_history_and_judging(tmp_path, monkeypatch):
    runner, config, services, clock = timed_run(tmp_path, monkeypatch)
    result = (await runner.evaluate(config, services=services))[0]
    assert not result.failures
    assert result.predictions[0].metadata["response_time_s"] == 5
    assert result.resource_metrics["common.response_mean_s"] == 5
    assert result.resource_metrics["common.judging_time_s"] == 40
    assert clock.value == 145


@pytest.mark.asyncio
async def test_agent_response_uses_same_timing_boundary(tmp_path, monkeypatch):
    runner, config, services, clock = timed_run(tmp_path, monkeypatch)
    async def agent(case, memory):
        used = await memory.search(case.query, scope=case.scope)
        clock.advance(8)
        return MemoryEpisode(output="answer", applied_memory=[used])
    result = (await runner.evaluate(config, agent=agent))[0]
    assert not result.failures
    assert result.resource_metrics["common.response_mean_s"] == 10
    assert result.resource_metrics["common.judging_time_s"] == 40


@pytest.mark.asyncio
async def test_learning_curve_and_resume_keep_original_response_times(tmp_path, monkeypatch):
    runner, config, services, clock = timed_run(tmp_path, monkeypatch)
    config = replace(config, protocol="experience_learning_curve", checkpoints=(0, 1))
    first = (await runner.evaluate(config, services=services))[0]
    assert not first.failures
    assert [point["resource_metrics"]["common.response_mean_s"] for point in first.learning_curve] == [5, 5]
    clock.value = 0
    resumed = (await runner.evaluate(replace(config, output_dir=str(tmp_path / "resumed")),
        services=services, resume_from=tmp_path / "results"))[0]
    assert not resumed.failures
    assert resumed.resource_metrics["common.response_mean_s"] == 5
    assert resumed.resource_metrics["common.response_timing_coverage"] == 1
    assert clock.value == 0
    assert runner.results(tmp_path / "resumed")[0].to_dict() == resumed.to_dict()


@pytest.mark.asyncio
async def test_memrl_sync_bridge_keeps_usage_attribution():
    from mematlas.methods.memrl.core import _run_async
    ledger = UsageLedger()
    async def complete():
        ledger.add(UsageRecord("fixture", "model", prompt_tokens=4))
        return "ok"
    with usage_scope(ledger, "memory"), usage_context(stage="training", operation="add", case_id="case"):
        assert _run_async(complete()) == "ok"
    record = ledger.all()[0]
    assert (record.stage, record.operation, record.case_id, record.phase) == ("training", "add", "case", "memory")


def test_memrl_storage_registration_follows_restored_databases(tmp_path):
    from mematlas.methods.memrl.method import MemRLMemory
    adapter = object.__new__(MemRLMemory)
    adapter.services = MethodServices()
    adapter._data_paths = set()
    native = SimpleNamespace(_cube_dir=tmp_path / "old-cube", _qdrant_dir=tmp_path / "old-db")
    adapter._instances = {"scope": native}
    adapter._refresh_data_paths()
    old_paths = set(adapter.services.resource_paths)
    native._cube_dir, native._qdrant_dir = tmp_path / "current-cube", tmp_path / "current-db"
    adapter._refresh_data_paths()
    assert not old_paths & adapter.services.resource_paths
    assert adapter.services.resource_paths == {(tmp_path / "current-cube").resolve(), (tmp_path / "current-db").resolve()}


def test_report_gives_academic_units_and_keeps_diagnostics(tmp_path):
    result = EvaluationResult("fixture", "data", (), {}, {
        "common.response_mean_s": 2, "common.memory_storage_mib": 3,
        "common.process_peak_rss_mib": 4, "common.memory_llm_calls": 5,
        "common.case_coverage": 1, "common.cost": 0.1,
    })
    ReportWriter().write([result], tmp_path)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Mean question response time (s/question)" in report
    assert "Live memory data storage | MiB | 3" in report
    assert "Memory LLM requests | requests | 5" in report
    assert "Run Diagnostics" in report and "common.case_coverage" in report
    assert "common.cost" not in report


@pytest.mark.asyncio
async def test_hospital_response_timing_survives_scoring_retry(tmp_path, monkeypatch):
    from mematlas.integrations.openhospital.execution import HospitalExecution
    from mematlas.test.evaluation.test_openhospital_adaptation import bridge_for, REPORT, FINAL
    clock, executions, scores = Clock(), [], []
    monkeypatch.setattr("mematlas.integrations.openhospital.execution.perf_counter", clock.now)
    bridge, _ = bridge_for(tmp_path / "memory")
    async def run_test_async(**options):
        executions.append(True)
        clock.advance(7)
        return {"final_results": [FINAL], "cases": [{"finished": True}], "output_dir": "sdk"}
    monkeypatch.setitem(sys.modules, "hospital_agent_sdk", SimpleNamespace(run_test_async=run_test_async))
    async def score(*args, **options):
        scores.append(True)
        clock.advance(11)
        if len(scores) == 1:
            raise RuntimeError("judge unavailable")
        return REPORT
    agent = SimpleNamespace(actions=SimpleNamespace(evaluation=score))
    execution = HospitalExecution(plan=None, output_dir=tmp_path / "run", source_dir=None,
        agent_class=lambda **kwargs: agent, doctor_config=None,
        runtime_builder=lambda *args, **kwargs: (None, None), service_urls={"primary_test": "url"}, team_id="team")
    async def no_service(*args):
        pass
    monkeypatch.setattr(execution, "_service_usage", no_service)
    monkeypatch.setattr(execution, "_bind_metered", lambda *args: None)
    monkeypatch.setattr(execution, "_scoring_agent", lambda *args: agent)
    try:
        with pytest.raises(RuntimeError, match="judge unavailable"):
            await execution.test_batch(["patient"], bridge, 0, "primary_test")
        result = await execution.test_batch(["patient"], bridge, 0, "primary_test")
        assert len(executions) == 1 and len(scores) == 2
        assert result["resource_metrics"]["common.response_mean_s"] == 7
        assert result["case_reports"][0]["response_time_s"] == 7
        assert Evaluator._operation_resources(bridge.core_memory.traces)["common.judging_time_s"] == 22
        cached = await execution.test_batch(["patient"], bridge, 0, "primary_test")
        assert cached["resource_metrics"]["common.response_mean_s"] == 7
        assert len(scores) == 2
    finally:
        await bridge.close()
