"""Hospital provenance, explicit metrics and verified recovery behavior."""
import sys
from types import SimpleNamespace
from dataclasses import replace
from pathlib import Path

import pytest

from mematlas import BatchResult, ExperimentConfig, ExperimentRunner, MemoryContribution
from mematlas.core.files import atomic_write_json
from mematlas.core.provenance import json_sha256
from mematlas.evaluation.experience import _ExperiencePlan
from mematlas.evaluation.metrics import Metric, default_metric_registry
from mematlas.integrations.openhospital.execution import HospitalExecution
from mematlas.integrations.openhospital.rsi import _HospitalEvaluation
from mematlas.test.evaluation.support import registry_for
from mematlas.test.evaluation.test_openhospital_adaptation import bridge_for, FINAL, REPORT, STATE


async def install_retrieval(bridge, plugin):
    await bridge.start()

    async def retrieve(*args, **kwargs):
        return MemoryContribution(
            items=[{"memory_id": "used", "content": "source", "source_event_ids": ["event"]},
                   {"memory_id": "unused", "content": "other"}],
            context=[{"content": "formatted source", "source_item_indices": [0]},
                     {"content": "other source", "source_item_indices": [1]}],
            extensions={"fixture": {"association": "preserved"}})

    plugin.instances[0].search = retrieve


@pytest.mark.asyncio
async def test_heldout_sources_require_confirmation_and_do_not_change_frozen_state(tmp_path):
    bridge, plugin = bridge_for(tmp_path / "memory", max_contexts=1)
    await install_retrieval(bridge, plugin)
    before = await bridge.state_fingerprint()
    try:
        async with bridge.core_memory.read_only():
            bridge._begin_test_case("patient")
            await bridge.retrieve(patient_id="patient", phase="candidate", mode="test")
            assert bridge._test_contributions("patient") == []
            assert await bridge.retrieve(patient_id="patient", phase="used", mode="test") == ["formatted source"]
            bridge.confirm_application("patient", phase="used")
            bridge.confirm_application("patient", phase="used")
            used = bridge._test_contributions("patient")
            assert len(used) == 1
            assert [item["memory_id"] for item in used[0]["items"]] == ["used"]
            assert used[0]["extensions"]["fixture"]["association"] == "preserved"
            assert await bridge.state_fingerprint() == before
        assert await bridge.state_fingerprint() == before
        assert not bridge.history.results and not plugin.instances[0].episodes
    finally:
        await bridge.close()


def execution_for(tmp_path, bridge, monkeypatch, *, run, score, source=None):
    monkeypatch.setitem(sys.modules, "hospital_agent_sdk", SimpleNamespace(run_test_async=run))
    agent = SimpleNamespace(actions=SimpleNamespace(evaluation=score))
    execution = HospitalExecution(plan=None, output_dir=tmp_path, source_dir=source,
        agent_class=lambda **kwargs: agent, doctor_config=None,
        runtime_builder=lambda *args, **kwargs: (None, None),
        service_urls={"primary_test": "url"}, team_id="team")

    async def no_service(*args):
        pass

    monkeypatch.setattr(execution, "_service_usage", no_service)
    monkeypatch.setattr(execution, "_bind_metered", lambda *args: None)
    monkeypatch.setattr(execution, "_scoring_agent", lambda *args: agent)
    return execution


@pytest.mark.asyncio
async def test_verified_execution_preserves_used_sources_through_scoring_retry_and_resume(tmp_path, monkeypatch):
    bridge, plugin = bridge_for(tmp_path / "memory", max_contexts=1)
    await install_retrieval(bridge, plugin)
    calls, scores = [], []

    async def run(**kwargs):
        calls.append(True)
        await bridge.retrieve(patient_id="patient", phase="diagnosis", mode="test")
        bridge.confirm_application("patient", phase="diagnosis")
        return {"final_results": [FINAL], "cases": [{"finished": True}], "output_dir": "sdk"}

    async def score(*args, **kwargs):
        scores.append(True)
        if len(scores) == 1:
            raise RuntimeError("judge unavailable")
        return REPORT

    execution = execution_for(tmp_path / "first", bridge, monkeypatch, run=run, score=score)
    try:
        async with bridge.core_memory.read_only():
            with pytest.raises(RuntimeError, match="judge unavailable"):
                await execution.test_batch(["patient"], bridge, 0, "primary_test")
            scored = await execution.test_batch(["patient"], bridge, 0, "primary_test")
        assert len(calls) == 1 and len(scores) == 2
        row = scored["case_reports"][0]
        assert row["applied_memory"][0]["items"][0]["memory_id"] == "used"
        assert row["final_result"] == FINAL
        recovered = execution_for(tmp_path / "second", bridge, monkeypatch, run=run, score=score,
                                  source=tmp_path / "first")
        async with bridge.core_memory.read_only():
            result = await recovered.test_batch(["patient"], bridge, 0, "primary_test")
        assert result["case_reports"] == scored["case_reports"]
        assert len(calls) == 1 and len(scores) == 2
    finally:
        await bridge.close()


@pytest.mark.asyncio
async def test_mutating_execution_is_never_cached_or_accepted_after_retry(tmp_path, monkeypatch):
    bridge, plugin = bridge_for(tmp_path / "memory")
    await bridge.start()
    calls = []

    async def run(**kwargs):
        calls.append(True)
        if len(calls) == 1:
            plugin.instances[0].episodes.append({"illegal": "write"})
        return {"final_results": [FINAL], "cases": [{"finished": True}], "output_dir": "sdk"}

    async def score(*args, **kwargs):
        return REPORT

    execution = execution_for(tmp_path / "run", bridge, monkeypatch, run=run, score=score)
    try:
        with pytest.raises(Exception, match="frozen memory|Frozen evaluation"):
            async with bridge.core_memory.read_only():
                await execution.test_batch(["patient"], bridge, 0, "primary_test")
        assert not list((tmp_path / "run").glob("test_execution/*.json"))
        async with bridge.core_memory.read_only():
            result = await execution.test_batch(["patient"], bridge, 0, "primary_test")
        assert len(calls) == 2 and result["diagnosis_accuracy"] == 1
    finally:
        await bridge.close()


@pytest.mark.asyncio
async def test_unverified_hospital_cache_is_reexecuted(tmp_path, monkeypatch):
    bridge, _ = bridge_for(tmp_path / "memory")
    calls = []

    async def run(**kwargs):
        calls.append(True)
        return {"final_results": [FINAL], "cases": [{"finished": True}], "output_dir": "new"}

    async def score(*args, **kwargs):
        return REPORT

    execution = execution_for(tmp_path / "run", bridge, monkeypatch, run=run, score=score)
    key = f"0-primary_test-{json_sha256('patient')}.json"
    atomic_write_json(execution.output / "test_execution" / key,
                      {"final_result": {"diagnosis": ["unverified"]}, "output_dir": "old"})
    atomic_write_json(execution.output / "test_cases" / key,
                      {"patient_id": "patient", "department": "medicine", "report": REPORT})
    try:
        async with bridge.core_memory.read_only():
            result = await execution.test_batch(["patient"], bridge, 0, "primary_test")
        assert calls == [True]
        assert result["case_reports"][0]["final_result"] == FINAL
    finally:
        await bridge.close()


@pytest.mark.asyncio
async def test_mutating_scoring_preserves_only_the_verified_execution(tmp_path, monkeypatch):
    bridge, plugin = bridge_for(tmp_path / "memory")
    await bridge.start()
    calls, scores = [], []

    async def run(**kwargs):
        calls.append(True)
        return {"final_results": [FINAL], "cases": [{"finished": True}], "output_dir": "sdk"}

    async def score(*args, **kwargs):
        scores.append(True)
        if len(scores) == 1:
            plugin.instances[0].episodes.append({"invalid": "judge write"})
        return REPORT

    execution = execution_for(tmp_path / "run", bridge, monkeypatch, run=run, score=score)
    try:
        with pytest.raises(Exception, match="frozen memory|Frozen evaluation"):
            async with bridge.core_memory.read_only():
                await execution.test_batch(["patient"], bridge, 0, "primary_test")
        assert list((tmp_path / "run").glob("test_execution/*.json"))
        assert not list((tmp_path / "run").glob("test_cases/*.json"))
        async with bridge.core_memory.read_only():
            result = await execution.test_batch(["patient"], bridge, 0, "primary_test")
        assert len(calls) == 1 and len(scores) == 2
        assert result["diagnosis_accuracy"] == 1
    finally:
        await bridge.close()


class AppliedClinicalMetric(Metric):
    name = "test.applied_clinical"

    def compute(self, predictions):
        return sum(bool(p.applied_memory) and p.output == FINAL
                   and p.metadata["clinical_report"]["diagnosisAccuracy"] == 1 for p in predictions) / len(predictions)


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.asyncio
async def test_hospital_public_results_run_explicit_metrics_and_keep_provenance(tmp_path, explicit):
    bridge, plugin = bridge_for(tmp_path / "memory")
    plan = _ExperiencePlan(("train",), ("heldout",), (), batch_count=1,
                           checkpoints=(0, 1), checkpoint_basis="completed")
    used = MemoryContribution(items=[{"memory_id": "confirmed", "content": "clinical source"}])
    execution = SimpleNamespace(on_case_completed=None, _find=lambda path: None)

    async def train(ids, owner, index):
        owner.configure_training_batch(ids, batch_index=index)
        await owner.learn_case(patient_id="train", reflection={}, case_state=STATE,
                               final_result=FINAL, evaluation_report=REPORT)
        await execution.on_case_completed(1)
        return BatchResult((owner.history.result(owner._occurrence("train"), scope=owner.scope),))

    async def test(ids, owner, checkpoint, split):
        return {"case_reports": [{"patient_id": "heldout", "department": "medicine", "report": REPORT,
                                   "final_result": FINAL, "applied_memory": [used.to_dict()]}]}

    execution.train_batch, execution.test_batch = train, test
    metric_registry = default_metric_registry()
    metric_registry.register(AppliedClinicalMetric)
    runner = ExperimentRunner(registry=registry_for(plugin), metric_registry=metric_registry)
    config = ExperimentConfig((bridge.core_memory._requested_config,), {"kind": "openhospital"},
        protocol="experience_learning_curve", checkpoints=(0, 1),
        metrics=(AppliedClinicalMetric.name,) if explicit else (), output_dir=str(tmp_path / "run"))
    result = (await runner.evaluate(config, agent=_HospitalEvaluation(plan, bridge, execution,
                                      execution_metadata={"case_resume": True})))[0]
    assert not result.failures
    assert result.metrics["diagnosis_accuracy"] == 1
    assert result.predictions[0].applied_memory == (used,)
    assert "common.exact_match" not in result.metrics and "common.token_f1" not in result.metrics
    if explicit:
        assert result.metrics[AppliedClinicalMetric.name] == 1
        assert all(point[AppliedClinicalMetric.name] == 1 for point in result.learning_curve)
    else:
        assert AppliedClinicalMetric.name not in result.metrics
    assert runner.results(tmp_path / "run")[0].to_dict() == result.to_dict()


@pytest.mark.asyncio
async def test_builtin_doctor_confirms_test_memory_notes(tmp_path, monkeypatch):
    import importlib
    pytest.importorskip("flask")
    runtime = Path(__file__).resolve().parents[2] / "integrations" / "openhospital_runtime"
    monkeypatch.syspath_prepend(str(runtime / "hospital-agent-sdk"))
    doctor_module = importlib.import_module(
        "mematlas.integrations.openhospital_runtime.hospital_agent_example.agent.agent")
    bridge, plugin = bridge_for(tmp_path / "memory", max_contexts=1)
    await install_retrieval(bridge, plugin)
    doctor = object.__new__(doctor_module.MyDoctorAgent)
    doctor.memory = bridge
    try:
        async with bridge.core_memory.read_only():
            bridge._begin_test_case("heldout")
            notes = await doctor._memory_notes({"patient_id": "heldout", "mode": "test"}, phase="diagnosis")
        assert notes == ["formatted source"]
        assert bridge._test_contributions("heldout")[0]["items"][0]["memory_id"] == "used"
    finally:
        await bridge.close()


@pytest.mark.asyncio
async def test_snapshot_method_restores_local_retrieval_changes_before_caching_each_case(tmp_path, monkeypatch):
    from mematlas.test.evaluation.support import FixturePlugin
    plugin = FixturePlugin(required_feedback=True)
    plugin.descriptor = replace(plugin.descriptor,
        capabilities=replace(plugin.descriptor.capabilities, read_only_mode="snapshot"))
    bridge, _ = bridge_for(tmp_path / "memory", plugin=plugin)
    await bridge.start()
    method = plugin.instances[0]
    baseline = await bridge.state_fingerprint()
    calls = []

    async def run(**kwargs):
        calls.append(True)
        method.episodes.append({"local_retrieval_cache": len(calls)})
        return {"final_results": [FINAL], "cases": [{"finished": True}], "output_dir": "sdk"}

    async def score(*args, **kwargs):
        method.episodes.append({"local_scoring_cache": True})
        return REPORT

    execution = execution_for(tmp_path / "run", bridge, monkeypatch, run=run, score=score)
    try:
        async with bridge.core_memory.read_only():
            result = await execution.test_batch(["one", "two"], bridge, 0, "primary_test")
            assert await bridge.state_fingerprint() == baseline
        assert result["evaluated_cases"] == 2 and len(calls) == 2
        async with bridge.core_memory.read_only():
            cached = await execution.test_batch(["one", "two"], bridge, 0, "primary_test")
        assert cached["case_reports"] == result["case_reports"] and len(calls) == 2
        assert await bridge.state_fingerprint() == baseline
    finally:
        await bridge.close()
