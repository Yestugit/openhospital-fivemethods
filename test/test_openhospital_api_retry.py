"""Offline API outage tests: preserve payloads, attempt accounting and recovery."""
from contextlib import nullcontext
from io import BytesIO
import json
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from mematlas.integrations.openhospital.application import load_doctor_runtime
from mematlas.core.telemetry import UsageLedger


@pytest.mark.parametrize("phase", ["train", "test"])
def test_doctor_survives_more_than_two_timeouts_without_changing_decision(monkeypatch, phase):
    load_doctor_runtime()
    from hospital_agent_sdk import llm
    bodies, sleeps, events = [], [], []
    def request(req, **kwargs):
        bodies.append(req.data)
        if len(bodies) <= 4:
            raise TimeoutError("temporary read timeout")
        data = {"content": "ok", "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1}}
        return nullcontext(SimpleNamespace(read=lambda: json.dumps(data).encode()))
    monkeypatch.setattr(llm.urllib.request, "urlopen", request)
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    caller = llm.LLMCaller("offline", phase=phase, team_id="t",
                           service_base_url="http://localhost", service_token="offline")
    caller.usage_observer = events.append
    assert caller._chat_sync(messages=[{"role": "user", "content": "same evidence"}],
                             temperature=.2, extra_body=None).content == "ok"
    assert len(bodies) == 5 and len(set(bodies)) == 1
    assert sleeps == [2, 4, 8, 16]
    assert [e["status"] for e in events] == ["failed"] * 4 + ["success"]


@pytest.mark.parametrize("status,attempts,exception", [(401, 1, RuntimeError), (429, 8, "paused"), (503, 8, "paused")])
def test_doctor_outage_pauses_but_invalid_credentials_fail_fast(monkeypatch, status, attempts, exception):
    load_doctor_runtime()
    from hospital_agent_sdk import llm
    events = []
    def request(req, **kwargs):
        raise HTTPError(req.full_url, status, "offline fixture", {}, BytesIO(b"offline fixture"))
    monkeypatch.setattr(llm.urllib.request, "urlopen", request)
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)
    caller = llm.LLMCaller("offline")
    caller.usage_observer = events.append
    with pytest.raises(llm.LLMRequestPaused if exception == "paused" else exception):
        caller._chat_sync(messages=[], temperature=.2, extra_body=None)
    assert len(events) == attempts
    assert all(e["status"] == "failed" for e in events)
    assert all(e['http_status']==status and e['elapsed_s']>=0 for e in events)
    assert len({e['request_id'] for e in events})==1


def test_doctor_respects_retry_after_and_preserves_request_identity(monkeypatch):
    load_doctor_runtime()
    from hospital_agent_sdk import llm
    sleeps=[];events=[]
    def request(req, **kwargs):
        if not events:
            raise HTTPError(req.full_url,429,'rate limit',{'Retry-After':'40'},BytesIO(b'limited'))
        return nullcontext(SimpleNamespace(read=lambda:json.dumps({'choices':[{'message':{'content':'{}'}}],
            'usage':{'prompt_tokens':2,'completion_tokens':1}}).encode()))
    monkeypatch.setattr(llm.urllib.request,'urlopen',request)
    monkeypatch.setattr(llm.time,'sleep',sleeps.append)
    caller=llm.LLMCaller('fixture');caller.usage_observer=events.append
    caller._chat_sync(messages=[],temperature=.2,extra_body=None)
    assert sleeps==[40] and [e['http_status'] for e in events]==[429,200]
    assert len({e['request_id'] for e in events})==1


@pytest.mark.asyncio
async def test_memory_retry_accounts_each_attempt_and_exhaustion_is_resumable(monkeypatch):
    from mematlas.providers import requests
    sleeps = []
    async def sleep(seconds): sleeps.append(seconds)
    monkeypatch.setattr(requests.asyncio, "sleep", sleep)
    ledger = UsageLedger()
    async def request(): raise TimeoutError("offline")
    with pytest.raises(requests.LLMRequestPaused):
        await requests.metered_request(request, ledger=ledger, provider="offline", model="fixture", kind="llm")
    assert ledger.summary()["failed_calls"] == 8
    assert sleeps == [2, 4, 8, 16, 30, 30, 30]


@pytest.mark.asyncio
async def test_api_pause_has_distinct_worker_exit_code(monkeypatch):
    from mematlas.integrations.openhospital import suite
    async def evaluate(*args, **kwargs):
        return [SimpleNamespace(failures=[{"error_type": "LLMRequestPaused"}])]
    monkeypatch.setattr(suite.ExperimentRunner, "evaluate", evaluate)
    monkeypatch.setattr(suite.ExperimentConfig, "load", lambda _: SimpleNamespace())
    monkeypatch.setattr(suite, "embedding_ledger", lambda _: UsageLedger())
    assert await suite.worker(SimpleNamespace(config="offline", source="prior-checkpoint")) == 76


def test_recovery_is_bounded_at_same_patient_and_resets_after_progress(monkeypatch):
    from mematlas.evaluation import artifacts
    from mematlas.integrations.openhospital.suite import queue_api_recovery
    position = 35
    monkeypatch.setattr(artifacts, "recovery_checkpoint", lambda _: (position, None, None))
    job = {"output": "offline", "status": "paused"}
    for expected in [30, 60, 120]:
        assert queue_api_recovery(job, 100)
        assert job["retry_after"] == 100 + expected
    job["status"] = "paused"
    assert not queue_api_recovery(job, 100)
    assert job["status"] == "paused"
    position = 36
    assert queue_api_recovery(job, 100) and job["api_recoveries"] == 1


@pytest.mark.asyncio
async def test_launcher_recovers_even_when_its_only_worker_pauses(tmp_path, monkeypatch):
    from contextlib import asynccontextmanager
    from pathlib import Path
    from mematlas.integrations.openhospital import suite
    from mematlas.evaluation import artifacts
    config = suite.make_experiment(tmp_path, "repeated", embedding="api")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    manifest = {"model_environment": {}, "smoke": False,
        "jobs": {"repeated": {"status": "paused", "attempt": 1, "config": str(config_path),
                               "output": config.output_dir, "source": None}}}
    (tmp_path / "suite.json").write_text(json.dumps(manifest), encoding="utf-8")
    starts = []
    async def spawn(*args, **kwargs):
        starts.append(args)
        return SimpleNamespace(returncode=76 if len(starts) == 1 else 0, pid=99999999)
    async def providers(_): return {"ready": True}
    @asynccontextmanager
    async def services(_): yield
    def next_attempt(run, condition, job):
        job.update(attempt=job["attempt"] + 1, status="pending", source="durable-35")
    queue = suite.queue_api_recovery
    monkeypatch.setattr(suite, "queue_api_recovery", lambda job, now: queue(job, now - 130))
    monkeypatch.setattr(artifacts, "recovery_checkpoint", lambda _: (35, None, None))
    monkeypatch.setattr(suite, "next_attempt", next_attempt)
    monkeypatch.setattr(suite, "model_environment", lambda: {})
    monkeypatch.setattr(suite, "check_providers", providers)
    monkeypatch.setattr(suite, "OpenHospitalServices", services)
    monkeypatch.setattr(suite, "print_status", lambda _: None)
    monkeypatch.setattr(suite.asyncio, "create_subprocess_exec", spawn)
    args = SimpleNamespace(command="resume", run_dir=str(tmp_path), stop_targets=None)
    assert await suite._run_suite(args) == 0
    final = json.loads((tmp_path / "suite.json").read_text(encoding="utf-8"))
    assert len(starts) == 2 and final["jobs"]["repeated"]["status"] == "completed"
    assert all("durable-35" in command for command in starts)
