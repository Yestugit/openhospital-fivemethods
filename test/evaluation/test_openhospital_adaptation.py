from dataclasses import replace
import json
import pytest
from mematlas import MemoryContribution, MemoryEpisode, MemoryEvent
from mematlas.integrations.openhospital import OpenHospitalMemoryIntegration
from mematlas.integrations.openhospital.rsi import _HospitalRunner
from mematlas.evaluation.experience import _ExperiencePlan
from mematlas.test.evaluation.support import FixturePlugin, config_for, registry_for

REPORT={"diagnosisAccuracy":1.0,"examinationPrecision":0.5,"treatmentOverallScore":0.9}
STATE={"chat_history":[{"from":"patient","text":"chest pain"}],
       "timeline":[{"kind":"message","role":"user","text":"chest pain"},
                   {"kind":"tool_result","role":"tool","text":"ECG result"}]}
FINAL={"diagnosis":["clinical diagnosis"]}


def bridge_for(path, *, plugin=None, **kwargs):
    plugin=plugin or FixturePlugin(required_feedback=True)
    config=config_for(path)
    config["method"]["method_id"]=plugin.descriptor.method_id
    return OpenHospitalMemoryIntegration.from_config(config,registry=registry_for(plugin),**kwargs),plugin


@pytest.mark.asyncio
async def test_ordered_doctor_cache_preserves_full_memory_and_task_selection():
    from types import SimpleNamespace
    from mematlas.integrations.openhospital.application import load_doctor_runtime
    agent_class, _ = load_doctor_runtime()
    from agent.prompt import format_prompt, NEXT_ACTION_PROMPT, DOCTOR_SYSTEM_PROMPT
    prompt = format_prompt(NEXT_ACTION_PROMPT, {"memory_notes": ["[Short-term conversation]\n" + "clinical evidence " * 700],
        "chat_history": [{"from": "patient", "text": "new evidence"}], "examinations": {"result": "normal"}})
    requests = []
    async def chat(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(content='{"action":"ask_patient"}', usage={})
    agent = agent_class(config={"prompt_cache": "ephemeral"})
    agent.llm = SimpleNamespace(chat=chat)
    for phase in ("next_action", "department", "evaluation_reflection"):
        await agent._llm_text(prompt, system_prompt=DOCTOR_SYSTEM_PROMPT,
            temperature=0.2, prompt_name=phase, patient_id="p")
    for request in requests:
        messages = request["messages"]
        assert messages[0] == {"role": "system", "content": DOCTOR_SYSTEM_PROMPT}
        shown = "".join(row["content"] if isinstance(row["content"], str)
                        else "".join(block["text"] for block in row["content"]) for row in messages[1:])
        assert shown == prompt  # Every character and every source stays in order.
        assert request["temperature"] == 0.2
    assert len(requests[0]["messages"]) == 3
    assert requests[0]["messages"][1]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert all(len(request["messages"]) == 3 for request in requests[1:])
    agent.config["prompt_cache"] = None
    await agent._llm_text(prompt, system_prompt=DOCTOR_SYSTEM_PROMPT,
        temperature=0.2, prompt_name="next_action", patient_id="p")
    assert requests[-1]["messages"][1]["content"] == prompt


@pytest.mark.parametrize("phase", ["train", "test"])
def test_ordered_cache_boundary_survives_direct_and_heldout_proxy(phase, monkeypatch):
    from mematlas.integrations.openhospital.application import load_doctor_runtime
    from mematlas.integrations.openhospital.config import OPENHOSPITAL_ROOT
    from contextlib import nullcontext
    from types import SimpleNamespace
    load_doctor_runtime()
    monkeypatch.syspath_prepend(str(OPENHOSPITAL_ROOT / "hospital_service"))
    from agent.prompt import llm_messages
    from hospital_agent_sdk.llm import LLMCaller
    from hospital_service.service.doctor_llm import _messages
    prompt = 'Task\n\n记忆摘要：\n' + json.dumps(["[Short-term conversation]\n" + "complete clinical memory " * 300]) + '\n\n历史对话：\n[]\n'
    messages = llm_messages("fixed system", prompt, cache_prefix=True)
    def respond(request, **kwargs):
        payload = json.loads(request.data)
        transmitted = _messages(payload["messages"])
        assert transmitted == messages
        assert "".join(block["text"] for row in transmitted[1:] for block in row["content"]) == prompt
        usage = {"prompt_tokens": 2000, "completion_tokens": 1,
                 "prompt_tokens_details": {"cached_tokens": 1500}}
        reply = ({"content": "{}", "usage": usage} if phase == "test" else
                 {"choices": [{"message": {"content": "{}"}}], "usage": usage})
        return nullcontext(SimpleNamespace(read=lambda: json.dumps(reply).encode()))
    monkeypatch.setattr("urllib.request.urlopen", respond)
    caller = LLMCaller("test-key", phase=phase, team_id="t",
                       service_base_url="http://localhost", service_token="token")
    result = caller._chat_sync(messages=messages, temperature=0.2, extra_body=None)
    assert result.usage["prompt_tokens_details"]["cached_tokens"] == 1500


@pytest.mark.parametrize("phase", ["train", "test"])
def test_cache_accounting_preserves_direct_and_proxy_request_messages(phase, monkeypatch):
    from mematlas.integrations.openhospital.application import load_doctor_runtime
    from mematlas.integrations.openhospital.config import OPENHOSPITAL_ROOT
    from mematlas.integrations.openhospital.accounting import doctor_usage_observer
    from mematlas.core.telemetry import UsageLedger
    from contextlib import nullcontext
    from types import SimpleNamespace
    load_doctor_runtime()
    monkeypatch.syspath_prepend(str(OPENHOSPITAL_ROOT / "hospital_service"))
    from hospital_agent_sdk.llm import LLMCaller
    from hospital_service.service.doctor_llm import _messages
    from hospital_service.service.llm_client import OpenAICompatibleChatClient
    from hospital_service.service.usage_events import CallUsageStore, record_model_call
    details = {"cached_tokens": 1500, "cache_creation_input_tokens": 0}
    usage = {"prompt_tokens": 2000, "completion_tokens": 1, "prompt_tokens_details": details}
    store, ledger = CallUsageStore(), UsageLedger()
    original_messages = [{"role": "system", "content": "system"},
        {"role": "user", "content": 'task\n\n记忆摘要：\n["complete evidence"]\n\n历史对话：\n[]'}]
    def respond(request, **kwargs):
        data = json.loads(request.data)
        messages = _messages(data["messages"])
        assert messages == original_messages
        assert "cache_control" not in request.data.decode("utf-8")
        if phase == "test":
            with store.scope(dataset_key="test", team_id="t", component="doctor"):
                record_model_call(usage, model="qwen3.5-flash", endpoint="https://maas.example/v1")
            normalized = OpenAICompatibleChatClient._normalize_usage(None, usage)
            reply = {"content": "{}", "usage": normalized}
        else:
            reply = {"choices": [{"message": {"content": "{}"}}], "usage": usage}
        return nullcontext(SimpleNamespace(read=lambda: json.dumps(reply).encode()))
    monkeypatch.setattr("urllib.request.urlopen", respond)
    caller = LLMCaller("test-key", service_base_url="http://localhost", service_token="token",
                       team_id="t", phase=phase)
    caller.usage_observer = doctor_usage_observer(ledger)
    response = caller._chat_sync(messages=original_messages, temperature=0.2, extra_body=None)
    assert response.usage["prompt_tokens_details"] == details
    if phase == "test":
        assert not ledger.all()  # Proxy calls are counted at their service origin once.
        rows = store.records(dataset_key="test", team_id="t")
        assert rows[0]["metadata"]["prompt_tokens_details"] == details
    else:
        assert ledger.summary()["cached_prompt_tokens"] == 1500


@pytest.mark.asyncio
async def test_hospital_applies_only_confirmed_full_contribution(tmp_path):
    bridge,plugin=bridge_for(tmp_path/"memory",journal_dir=tmp_path/"encounters",max_contexts=1)
    await bridge.start()
    method=plugin.instances[0]
    async def retrieve(query,**kwargs):
        return MemoryContribution(items=[{"memory_id":"one","content":"first","source_event_ids":["s1"]},
                                         {"memory_id":"two","content":"second","source_event_ids":["s2"]}],
            context=[{"content":"formatted first","source_item_indices":[0]},
                     {"content":"formatted second","source_item_indices":[1]}],
            extensions={"fixture":{"query":"opaque association"}})
    method.search=retrieve
    bridge.configure_training_batch(["p"],batch_index=1)
    assert await bridge.retrieve(patient_id="p",phase="diagnosis") == ["formatted first"]
    bridge.confirm_application("p",phase="diagnosis")
    bridge.confirm_application("p",phase="diagnosis")
    await bridge.append_case_reflection(patient_id="p",case_state=STATE,final_result=FINAL,
        evaluation_report=REPORT,evaluation_reflection={"summary":"follow up"})
    learned=method.episodes[0]
    assert learned["task"].endswith("chest pain")
    assert learned["success"] is True and learned["reward"] == pytest.approx(0.8)
    assert len(learned["applied_memory"]) == 1
    used=learned["applied_memory"][0]
    assert [row["memory_id"] for row in used["items"]] == ["one"]
    assert used["context"][0]["source_item_indices"] == [0]
    assert used["extensions"]["fixture"]["query"] == "opaque association"
    assert bridge.reuse_statistics()["distinct_applied_memories"] == 1
    assert all(isinstance(row["content"],str) for row in learned["events"])
    assert learned["episode_id"] == "batch:1:patient:p"
    await bridge.close()


@pytest.mark.asyncio
async def test_hospital_unconfirmed_retrieval_is_not_credited_and_test_is_frozen(tmp_path):
    bridge,plugin=bridge_for(tmp_path/"memory")
    await bridge.start()
    async def retrieve(query,**kwargs):
        return MemoryContribution(items=[{"memory_id":"candidate","content":"unused"}],extensions={"fixture":{}})
    plugin.instances[0].search=retrieve
    await bridge.retrieve(patient_id="p",phase="diagnosis")
    await bridge.learn_case(patient_id="p",reflection={},case_state=STATE,final_result=FINAL,evaluation_report=REPORT)
    assert plugin.instances[0].episodes[0]["applied_memory"] == []
    before=await bridge.state_fingerprint()
    async with bridge.core_memory.read_only():
        await bridge.retrieve(patient_id="heldout",phase="diagnosis",mode="test")
        bridge.confirm_application("heldout",phase="diagnosis")
        await bridge.record_execution(patient_id="heldout",case_state=STATE,final_result=FINAL)
    assert await bridge.state_fingerprint() == before
    assert len(plugin.instances[0].episodes) == 1
    await bridge.close()


@pytest.mark.asyncio
async def test_delayed_clinical_feedback_restores_original_episode_and_scope(tmp_path):
    bridge,plugin=bridge_for(tmp_path/"source",feedback_timing="delayed",feedback_delay=1)
    bridge.configure_training_batch(["p"],batch_index=1)
    await bridge.learn_case(patient_id="p",reflection={},case_state=STATE,final_result=FINAL,evaluation_report=REPORT)
    assert not plugin.instances[0].episodes
    assert bridge.history.counts()["pending_feedback"] == 1
    await bridge.snapshot_to(tmp_path/"snapshot.json",learned_count=1)
    await bridge.close()
    restored,plugin=bridge_for(tmp_path/"restored",feedback_timing="delayed",feedback_delay=1)
    await restored.start()
    await restored.restore_from(tmp_path/"snapshot.json")
    restored.configure_training_batch(["p"],batch_index=2)
    await restored.learn_case(patient_id="p",reflection={},case_state=STATE,final_result=FINAL,evaluation_report=REPORT)
    assert len(plugin.instances[0].episodes) == 1
    assert plugin.instances[0].episodes[0]["episode_id"] == "batch:1:patient:p"
    assert plugin.instances[0].episodes[0]["scope"] == {"user_id":"alice"}
    assert restored.history.counts()["pending_feedback"] == 1
    await restored.close()


@pytest.mark.asyncio
async def test_batch_feedback_and_training_end_survive_snapshot(tmp_path):
    plugin=FixturePlugin(required_feedback=True)
    plugin.descriptor=replace(plugin.descriptor,method_id="memrl")
    plugin.qa_factory="mematlas.methods.memrl.qa:MemRLQA"
    bridge,_=bridge_for(tmp_path/"source",plugin=plugin,feedback_timing="batch")
    bridge.configure_training_batch(["p"],batch_index=1,end_of_training=True)
    await bridge.learn_case(patient_id="p",case_state=STATE,final_result=FINAL,
                            evaluation_report=REPORT,reflection={"summary":"real feedback"})
    assert not plugin.instances[0].episodes
    await bridge.snapshot_to(tmp_path/"snapshot.json",learned_count=1)
    await bridge.close()
    restored,_=bridge_for(tmp_path/"restored",plugin=plugin,feedback_timing="batch")
    await restored.start()
    await restored.restore_from(tmp_path/"snapshot.json")
    calls=[]
    async def feedback(bundle,owner,occurrence):
        calls.append(occurrence)
        return {**bundle,"evaluation_report":REPORT,"reflection":{"summary":"real feedback"}}
    restored.feedback_resolver=feedback
    await restored.deliver_feedback(batch_end=True)
    learned=plugin.instances[-1].episodes[0]
    assert calls == []
    assert learned["extensions"]["memrl"]["end_of_training"] is True
    assert learned["success"] is True and learned["feedback"]
    assert not restored.history.pending_feedback
    await restored.deliver_feedback(batch_end=True)
    assert len(plugin.instances[-1].episodes) == 1
    await restored.close()


@pytest.mark.asyncio
async def test_hospital_runner_preserves_completed_and_learned_counts(tmp_path):
    bridge,plugin=bridge_for(tmp_path/"memory")
    plan=_ExperiencePlan(("train",),("heldout",),(),batch_count=1,checkpoints=(0,1),checkpoint_basis="completed")
    async def train(ids,bridge,index):
        from mematlas import BatchResult
        bridge.configure_training_batch(ids,batch_index=index,end_of_training=True)
        await bridge.record_execution(patient_id="train",case_state=STATE,final_result=FINAL)
        assert bridge.history.result(bridge._occurrence("train"),scope=bridge.scope).execution == "completed"
        await bridge.learn_case(patient_id="train",reflection={},case_state=STATE,final_result=FINAL,evaluation_report=REPORT)
        return BatchResult(tuple(bridge.history.result(bridge._occurrence(p),scope=bridge.scope) for p in ids))
    async def test(ids,bridge,checkpoint,split):
        assert bridge.core_memory.is_read_only
        await bridge.retrieve(patient_id=ids[0],phase="diagnosis",mode="test")
        return {"diagnosis_accuracy":1.0}
    report=await _HospitalRunner(plan,bridge,tmp_path/"run",train,test).run()
    assert report["status"] == "completed"
    assert report["completed_exposures"] == report["learned_exposures"] == 1
    assert report["memory_reuse"]["learned_encounters"] == 1
    assert (tmp_path/"run/snapshots/learned_1.json").exists()
    assert not any(row.operation == "evolve" for row in bridge.core_memory.traces.all())


@pytest.mark.asyncio
async def test_invalid_clinical_feedback_does_not_learn(tmp_path):
    bridge,plugin=bridge_for(tmp_path/"memory")
    with pytest.raises(ValueError,match="Incomplete"):
        await bridge.learn_case(patient_id="p",reflection={},case_state=STATE,final_result=FINAL,
            evaluation_report={"diagnosisAccuracy":1.0})
    assert not plugin.instances[0].episodes and not bridge.history.pending_feedback
    await bridge.close()


@pytest.mark.asyncio
async def test_hospital_scene_uses_public_runner_and_unified_results(tmp_path):
    from types import SimpleNamespace
    from mematlas import ExperimentConfig, ExperimentRunner, BatchResult
    from mematlas.evaluation.experience import _ExperiencePlan
    from mematlas.integrations.openhospital.rsi import _HospitalEvaluation
    from mematlas.integrations.openhospital.metrics import aggregate_openhospital_evaluations
    bridge, plugin = bridge_for(tmp_path / "memory", feedback_timing="delayed", feedback_delay=1)
    plan = _ExperiencePlan(("one", "two"), ("heldout",), (), batch_count=2,
                           checkpoints=(0, 1, 2), checkpoint_basis="completed")
    calls = []
    execution = SimpleNamespace(on_case_completed=None,
        _find=lambda path: {"final_result": FINAL} if path.startswith("test_execution/") else None)
    async def train(ids, bridge, index):
        bridge.configure_training_batch(ids, batch_index=index, end_of_training=index == 2)
        offset = sum(len(batch) for batch in plan.batches()[:index - 1])
        for number, patient in enumerate(ids, start=1):
            calls.append(patient)
            await bridge.record_execution(patient_id=patient, case_state=STATE, final_result=FINAL)
            await bridge.learn_case(patient_id=patient, reflection={}, case_state=STATE,
                                    final_result=FINAL, evaluation_report=REPORT)
            await execution.on_case_completed(offset + number)
        return BatchResult(tuple(bridge.history.result(bridge._occurrence(p), scope=bridge.scope) for p in ids))
    async def test(ids, bridge, checkpoint, split):
        assert bridge.core_memory.is_read_only
        before = await bridge.state_fingerprint()
        await bridge.retrieve(patient_id=ids[0], phase="diagnosis", mode="test")
        assert await bridge.state_fingerprint() == before
        rows = [{"patient_id": p, "department": "心内科", "report": REPORT} for p in ids]
        return {**aggregate_openhospital_evaluations(rows), "case_reports": rows}
    execution.train_batch, execution.test_batch = train, test
    agent = _HospitalEvaluation(plan, bridge, execution, execution_metadata={"case_resume": True})
    config = ExperimentConfig((bridge.core_memory._requested_config,),
        {"kind": "openhospital", "path": "scene-roster"}, protocol="experience_learning_curve",
        checkpoints=(0, 1, 2), output_dir=str(tmp_path / "run"))
    runner = ExperimentRunner(registry=registry_for(plugin))
    assert runner.check(config, agent=agent)["ready"]
    result = (await runner.evaluate(config, agent=agent))[0]
    assert not result.failures
    assert calls == ["one", "two"]
    assert result.run_metadata["progress"]["completed_exposures"] == 2
    assert result.run_metadata["progress"]["learned_exposures"] == 1
    assert result.run_metadata["progress"]["pending_feedback"] == 1
    assert result.metrics["treatment_overall_score"] == 0.9
    assert result.predictions[0].output == FINAL
    assert [p["checkpoint"] for p in result.learning_curve] == [0, 1, 2]
    assert runner.results(tmp_path / "run")[0].to_dict() == result.to_dict()




@pytest.mark.asyncio
async def test_clinical_timestamp_is_durable_across_execution_and_learning(tmp_path):
    bridge, plugin = bridge_for(tmp_path / "memory", journal_dir=tmp_path / "encounters")
    bridge.configure_training_batch(["p"], batch_index=1)
    try:
        await bridge.record_execution(patient_id="p", case_state=STATE, final_result=FINAL)
        original = next(iter(bridge._bundles.values()))["case_state"]["encounter_timestamp"]
        await bridge.learn_case(patient_id="p", case_state=STATE, final_result=FINAL,
                                evaluation_report=REPORT, reflection={})
        assert next(iter(bridge._bundles.values()))["case_state"]["encounter_timestamp"] == original
        events = plugin.instances[0].episodes[0]["events"]
        assert all(row["timestamp"].startswith(original[:10]) for row in events)
        assert "encounter_timestamp" not in STATE
    finally:
        await bridge.close()


@pytest.mark.asyncio
async def test_training_only_recovery_and_bounded_snapshots(tmp_path):
    from mematlas import BatchResult
    from mematlas.integrations.openhospital.streams import _ExperienceStreamPlan
    from mematlas.evaluation.artifacts import recovery_checkpoint
    ids = tuple(f"p{i}" for i in range(6))
    plan = _ExperienceStreamPlan(ids, (), (), batch_count=6, training_only=True,
        checkpoints=(), training_batches=tuple((patient,) for patient in ids))
    calls = []
    async def train(patients, bridge, index):
        bridge.configure_training_batch(patients, batch_index=index)
        await bridge.learn_case(patient_id=patients[0], case_state=STATE, final_result=FINAL,
                                evaluation_report=REPORT, reflection={})
        calls.extend(patients)
        return BatchResult(tuple(bridge.history.result(bridge._occurrence(p), scope=bridge.scope) for p in patients))
    async def no_tests(*args):
        raise AssertionError("Training-only plans cannot execute held-out cases")
    metadata = {"snapshot_keep_recent": 2, "snapshot_keep_every": 3}
    bridge, _ = bridge_for(tmp_path / "memory")
    runner = _HospitalRunner(plan, bridge, tmp_path / "run", train, no_tests, execution_metadata=metadata)
    report = await runner.run()
    assert report["learned_exposures"] == 6 and report["primary_test_count"] == 0
    assert not report["unreached_checkpoints"] and plan.effective_checkpoints() == ()
    assert {p.stem for p in (tmp_path / "run/snapshots").glob("learned_*.json")} == {
        "learned_0", "learned_3", "learned_5", "learned_6"}
    assert len(list((tmp_path / "run/snapshots").glob("memory-*.json"))) == 4
    assert recovery_checkpoint(tmp_path / "run")[0] == 6
    restored, _ = bridge_for(tmp_path / "restored")
    resumed = _HospitalRunner(plan, restored, tmp_path / "resumed", train, no_tests,
        execution_metadata=metadata, resume_from=tmp_path / "run")
    await resumed.run()
    assert calls == list(ids)




def test_suite_recovers_a_job_that_failed_before_services_started(tmp_path):
    from mematlas import ExperimentConfig
    from mematlas.core.files import atomic_write_json
    from mematlas.integrations.openhospital.suite import make_experiment, next_attempt
    original = make_experiment(tmp_path, "repeated", smoke=True, embedding="api")
    path = tmp_path / "repeated/attempt-001/config.json"
    atomic_write_json(path, original.to_dict())
    job = {"config": str(path), "output": original.output_dir, "attempt": 1, "source": None}
    next_attempt(tmp_path, "repeated", job)
    restored = ExperimentConfig.load(job["config"])
    assert job["attempt"] == 2 and job["source"] is None
    assert restored.output_dir != original.output_dir
    assert restored.methods[0].runtime.artifact_dir != original.methods[0].runtime.artifact_dir
    assert restored.methods[0].scope == original.methods[0].scope
    from mematlas.integrations.openhospital.config import HospitalOptions
    assert HospitalOptions.from_experiment(restored) == HospitalOptions.from_experiment(original)


@pytest.mark.asyncio
async def test_failed_first_heldout_never_reports_training_scores_as_test_scores(tmp_path):
    from types import SimpleNamespace
    from mematlas import BatchResult, ExperimentConfig, ExperimentRunner
    from mematlas.integrations.openhospital.rsi import _HospitalEvaluation
    bridge, plugin = bridge_for(tmp_path / "memory")
    plan = _ExperiencePlan(("p",), ("heldout",), (), checkpoints=(1,), checkpoint_basis="completed")
    async def train(ids, bridge, index):
        bridge.configure_training_batch(ids, batch_index=index)
        await bridge.learn_case(patient_id="p", case_state=STATE, final_result=FINAL,
                                evaluation_report=REPORT, reflection={})
        return BatchResult((bridge.history.result(bridge._occurrence("p"), scope=bridge.scope),),
            {"case_reports": [{"patient_id": "p", "department": "blood", "report": REPORT}]})
    async def fail_test(*args):
        raise RuntimeError("judge unavailable")
    execution = SimpleNamespace(train_batch=train, test_batch=fail_test, _find=lambda path: None)
    agent = _HospitalEvaluation(plan, bridge, execution)
    config = ExperimentConfig((bridge.core_memory._requested_config,), {"kind": "openhospital"},
        protocol="experience_learning_curve", checkpoints=(1,), output_dir=str(tmp_path / "run"))
    result = (await ExperimentRunner(registry=registry_for(plugin)).evaluate(config, agent=agent))[0]
    assert result.failures and not result.metrics and not result.predictions
    assert result.run_metadata["metrics_split"] == "primary_test"
