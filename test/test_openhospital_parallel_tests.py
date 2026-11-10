"""Frozen test concurrency: real Mem0, provenance, accounting and recovery."""
from mematlas.evaluation.validation import submit_episode
import asyncio
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from mematlas import MemoryEpisode, MemoryEvent, MethodServices
from mematlas.integrations.openhospital.bridge import OpenHospitalMemoryIntegration
from mematlas.integrations.openhospital.execution import HospitalExecution
from mematlas.integrations.openhospital.progress import HospitalPaused
from mematlas.core.provenance import json_sha256
from mematlas.core.telemetry import current_usage_context
from mematlas.test.mem0.native_support import Vectors, require_native
from mematlas.test.evaluation.test_openhospital_adaptation import REPORT


async def make_bridge(path):
    require_native()
    bridge = OpenHospitalMemoryIntegration.from_config({
        "method": {"method_id": "mem0", "params": {"infer": False, "embedding_dims": 16,
            "threshold": 0, "write_policy": "episode"}},
        "scope": {"user_id": "parallel-test", "agent_id": "doctor"},
        "runtime": {"artifact_dir": str(path)}}, services=MethodServices(embedder=Vectors()))
    await bridge.start()
    await submit_episode(bridge.core_memory, MemoryEpisode(task="clinical source", episode_id="source",
        events=[MemoryEvent("message", "complete evidence", role="user")], scope=bridge.scope))
    return bridge


@dataclass
class Runtime:
    patient_ids: list
    output_dir: str


def execution(path, bridge, monkeypatch, *, split="primary_test", run=None, score=None, source=None):
    monkeypatch.setitem(sys.modules, "hospital_agent_sdk", SimpleNamespace(run_test_async=run))
    instance = HospitalExecution(plan=None, output_dir=path, source_dir=source,
        agent_class=lambda **kwargs: SimpleNamespace(memory=kwargs["memory"], actions=SimpleNamespace(evaluation=score)),
        doctor_config=None, runtime_builder=lambda *args, **kwargs: ({}, Runtime(kwargs["patient_ids"], str(path / "sdk"))),
        service_urls={split: "url"}, team_id="team")
    async def no_service(*args): pass
    monkeypatch.setattr(instance, "_service_usage", no_service)
    monkeypatch.setattr(instance, "_bind_metered", lambda agent, bridge: agent)
    monkeypatch.setattr(instance, "_scoring_agent", lambda *args: SimpleNamespace(actions=SimpleNamespace(evaluation=score)))
    return instance


@pytest.mark.asyncio
@pytest.mark.parametrize("split", ["primary_test", "advanced_test"])
async def test_parallel_requests_overlap_but_keep_roster_provenance_and_frozen_memory(tmp_path, monkeypatch, split):
    bridge = await make_bridge(tmp_path / "memory")
    ids = ["slow", "medium", "fast", "last"]
    active, peak, calls, outputs = 0, 0, [], []
    before = await bridge.state_fingerprint()
    async def run(agent_factory, config):
        nonlocal active, peak
        patient = config.patient_ids[0]
        calls.append(patient); outputs.append(config.output_dir)
        assert len(Path(config.output_dir).name) <= 12
        active += 1; peak = max(peak, active)
        assert bridge.core_memory.is_read_only
        assert current_usage_context()["case_id"] == f"50:{split}:{patient}"
        await bridge.retrieve(patient_id=patient, phase="diagnosis", mode="test")
        bridge.confirm_application(patient, phase="diagnosis")
        await asyncio.sleep(.04 if patient == "slow" else .01)
        active -= 1
        return {"final_results": [{"diagnosis": [patient]}], "cases": [{"finished": True}], "output_dir": config.output_dir}
    async def score(patient, **kwargs):
        assert current_usage_context()["case_id"] == f"50:{split}:{patient}"
        assert bridge.core_memory.is_read_only
        return REPORT
    instance = execution(tmp_path / "parallel", bridge, monkeypatch, split=split, run=run, score=score)
    try:
        monkeypatch.setenv("OPH_TEST_CONCURRENCY", "3")
        result = await instance.test_batch(ids, bridge, 50, split)
        assert peak == 3 and len(calls) == len(ids)
        assert len(set(outputs)) == len(ids)
        assert [r["patient_id"] for r in result["case_reports"]] == ids
        assert all(r["final_result"]["diagnosis"] == [r["patient_id"]] for r in result["case_reports"])
        assert all(r["applied_memory"][0]["items"][0]["content"] == "complete evidence" for r in result["case_reports"])
        assert await bridge.state_fingerprint() == before
        assert not bridge.history.results
        # Compare the same deterministic clinical inputs with sequential execution.
        monkeypatch.setenv("OPH_TEST_CONCURRENCY", "1")
        serial = execution(tmp_path / "serial", bridge, monkeypatch, split=split, run=run, score=score)
        comparison = await serial.test_batch(ids, bridge, 50, split)
        assert result["diagnosis_accuracy"] == comparison["diagnosis_accuracy"]
        assert [r["final_result"] for r in result["case_reports"]] == [r["final_result"] for r in comparison["case_reports"]]
        assert [r["applied_memory"] for r in result["case_reports"]] == [r["applied_memory"] for r in comparison["case_reports"]]
        assert await bridge.state_fingerprint() == before
    finally:
        await bridge.close()


@pytest.mark.asyncio
async def test_scoring_failure_drains_siblings_and_resume_reuses_executed_cases(tmp_path, monkeypatch):
    monkeypatch.setenv("OPH_TEST_CONCURRENCY", "3")
    bridge = await make_bridge(tmp_path / "memory")
    calls, scored = [], []
    async def run(agent_factory, config):
        patient = config.patient_ids[0]; calls.append(patient)
        await asyncio.sleep(.01)
        return {"final_results": [{"diagnosis": [patient]}], "cases": [{"finished": True}], "output_dir": config.output_dir}
    async def score(patient, **kwargs):
        scored.append(patient)
        if patient == "fail" and scored.count(patient) == 1:
            raise RuntimeError("judge unavailable")
        await asyncio.sleep(.02)
        return REPORT
    first = execution(tmp_path / "first", bridge, monkeypatch, run=run, score=score)
    ids = ["fail", "saved-1", "saved-2", "not-started"]
    try:
        with pytest.raises(RuntimeError, match="judge unavailable"):
            await first.test_batch(ids, bridge, 50, "primary_test")
        assert set(calls) == set(ids[:3])
        assert len(list((first.output / "test_cases").glob("*.json"))) == 2
        second = execution(tmp_path / "second", bridge, monkeypatch, run=run, score=score, source=first.output)
        result = await second.test_batch(ids, bridge, 50, "primary_test")
        assert len(result["case_reports"]) == 4
        assert calls == ids  # successful diagnosis never repeated, including failed scoring
        assert scored.count("saved-1") == scored.count("saved-2") == 1
        assert scored.count("fail") == 2
    finally:
        await bridge.close()


@pytest.mark.asyncio
async def test_service_collector_is_serialized_while_cases_overlap(tmp_path, monkeypatch):
    import time
    from mematlas.integrations.openhospital import execution as module
    active, peak, calls = 0, 0, []
    class Collector:
        def __init__(self, *args): pass
        def sync(self):
            nonlocal active, peak
            active += 1; peak = max(peak, active)
            time.sleep(.01)
            calls.append(True); active -= 1
    monkeypatch.setattr(module, "HospitalUsageCollector", Collector)
    instance = HospitalExecution(plan=None, output_dir=tmp_path, source_dir=None, agent_class=None,
        doctor_config=None, runtime_builder=None, service_urls={}, team_id="team")
    await asyncio.gather(*(instance._service_usage("same") for _ in range(5)))
    assert peak == 1 and len(calls) == 5 and len(instance._collectors) == 1


@pytest.mark.asyncio
async def test_stop_drains_active_cases_then_resumes_only_missing_patients(tmp_path, monkeypatch):
    monkeypatch.setenv("OPH_TEST_CONCURRENCY", "3")
    bridge = await make_bridge(tmp_path / "memory")
    stop = False
    calls = []
    class Progress:
        bridge = None
        def __init__(self): self.tests = {}; self.last_test = {}; self.active = []
        def check_limits(self): pass
        def check_stop(self, _):
            if stop: raise HospitalPaused("durable stop")
        def write(self, **kwargs): self.active.append(kwargs.get("active_test_count", 0))
        def record_checkpoint(self, *args): pass
        @staticmethod
        def metrics(rows): return {}
    async def run(agent_factory, config):
        nonlocal stop
        patient = config.patient_ids[0]; calls.append(patient)
        await asyncio.sleep(.01)
        if patient == "first": stop = True
        return {"final_results": [{"diagnosis": [patient]}], "cases": [{"finished": True}], "output_dir": config.output_dir}
    async def score(*args, **kwargs): return REPORT
    ids = ["first", "in-flight-1", "in-flight-2", "queued"]
    instance = execution(tmp_path / "first", bridge, monkeypatch, run=run, score=score)
    instance.progress = Progress()
    before = await bridge.state_fingerprint()
    try:
        with pytest.raises(HospitalPaused):
            await instance.test_batch(ids, bridge, 50, "primary_test")
        assert calls == ids[:3]
        assert len(instance.progress.tests) == 3
        assert instance.progress.active[-1] == 0 and max(instance.progress.active) == 3
        assert len(list((instance.output / "test_cases").glob("*.json"))) == 3
        stop = False
        resumed = execution(tmp_path / "resumed", bridge, monkeypatch, run=run, score=score, source=instance.output)
        await resumed.test_batch(ids, bridge, 50, "primary_test")
        assert calls == ids and await bridge.state_fingerprint() == before
    finally:
        await bridge.close()


@pytest.mark.asyncio
async def test_mutating_parallel_case_cannot_seal_results(tmp_path, monkeypatch):
    monkeypatch.setenv("OPH_TEST_CONCURRENCY", "2")
    bridge = await make_bridge(tmp_path / "memory")
    async def run(agent_factory, config):
        bridge.history.exposures += 1  # hostile simulated agent bypasses the facade
        await asyncio.sleep(.01)
        return {"final_results": [{"diagnosis": [config.patient_ids[0]]}],
                "cases": [{"finished": True}], "output_dir": config.output_dir}
    async def score(*args, **kwargs): return REPORT
    instance = execution(tmp_path / "out", bridge, monkeypatch, run=run, score=score)
    try:
        with pytest.raises(RuntimeError, match="frozen memory"):
            await instance.test_batch(["one", "two"], bridge, 50, "primary_test")
        assert not list((instance.output / "test_execution").glob("*.json"))
        assert not list((instance.output / "test_cases").glob("*.json"))
    finally:
        await bridge.close()
