"""Mem0 hospital contract, format fallback, cache and pilot regressions."""
import json
from collections import OrderedDict
from threading import Lock
from types import SimpleNamespace
import pytest
from mematlas.integrations.openhospital.application import load_doctor_runtime
from mematlas.integrations.openhospital.suite import make_experiment, make_pilot, CONDITIONS
from mematlas.integrations.openhospital.planning import build_plan
from mematlas.integrations.openhospital.config import HospitalOptions


def test_formal_1100_and_pilot_prefix(tmp_path):
    configs = [make_experiment(tmp_path, condition, embedding="api") for condition in CONDITIONS]
    counts=[]
    for config in configs:
        plan=build_plan(HospitalOptions.from_experiment(config)).plan
        counts.append(plan.training_count + len(plan.primary_test_ids)*len(plan.effective_checkpoints()) + len(plan.advanced_test_ids))
        assert config.methods[0].method.params == {"implementation_mode":"standard", "algorithm_profile":"additive",
            "infer":True,"write_policy":"episode","hybrid_search":True,"rerank":False,"top_k":5}
    assert counts == [250,250,600]
    # No local weights needed to construct the pilot plan in this unit test.
    from dataclasses import replace
    config=replace(configs[0], benchmark={**configs[0].benchmark,"training_only":False,
        "heldout_per_department":2,"advanced_per_department":0,"evaluation_departments":("血液科",),"pilot_training_count":20},checkpoints=(20,))
    pilot=build_plan(HospitalOptions.from_experiment(config),checkpoints=config.checkpoints).plan
    formal=build_plan(HospitalOptions.from_experiment(configs[0])).plan
    assert list(sum(pilot.batches(),())) == list(formal.batches()[0][:20])
    assert [len(batch) for batch in pilot.batches()] == [10,10]
    assert pilot.effective_checkpoints() == (20,)
    assert len(pilot.primary_test_ids) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("responses", [
    ["invalid"]*4,
    ['{"examinations":"CBC","reason":"x"}']*4,
    ['{"examinations":["outside"],"reason":"x"}']*4,
    ['{"examinations":[42],"reason":"x"}']*4,
])
async def test_exam_fallback_is_bounded_and_audited(tmp_path,responses):
    cls,_=load_doctor_runtime();agent=cls(config={"json_exam_fallback":True,"log_llm_prompts":True})
    calls=[];events=[];agent._timeline=[]
    async def chat(**kwargs):
        calls.append(kwargs);return SimpleNamespace(content=responses[len(calls)-1],usage={"prompt_tokens":2,"completion_tokens":1})
    agent.llm=SimpleNamespace(chat=chat,model="test")
    agent.logger=SimpleNamespace(output_dir=tmp_path,write_event=lambda **event:events.append(event))
    agent.runtime_settings=SimpleNamespace(team_id="test")
    result=await agent._call_llm(prompt="clinical state",default={"examinations":["CBC"],"reason":"fallback"},
        prompt_name="exam_item",patient_id="p",allowed_examinations=["CBC"],already_ordered=[])
    assert result["examinations"] == ["CBC"]
    assert len(calls)==4 and all(call["extra_body"]["response_format"]["type"]=="json_object" for call in calls)
    assert agent._timeline[0]["json_fallback"]
    assert len([e for e in events if e["event_type"]=="DOCTOR_LLM_USAGE"])==4
    assert len([e for e in events if e["event_type"]=="DOCTOR_JSON_INVALID"])==4
    assert events[-1]["event_type"]=="DOCTOR_JSON_FALLBACK"
    assert len(list((tmp_path/"llm_prompts").glob("*.txt")))==4


@pytest.mark.asyncio
async def test_exhausted_catalog_and_successful_repair(tmp_path):
    cls,_=load_doctor_runtime();agent=cls(config={"json_exam_fallback":True});agent._timeline=[]
    responses=iter(["invalid",'{"examinations":[],"reason":"no more items"}'])
    async def chat(**kwargs): return SimpleNamespace(content=next(responses),usage={})
    agent.llm=SimpleNamespace(chat=chat)
    result=await agent._call_llm(prompt="state",default={"examinations":[],"reason":""},
        prompt_name="exam_item",allowed_examinations=[],already_ordered=[])
    assert result["examinations"]==[] and not agent._timeline[0]["json_fallback"]


def test_exact_embedding_cache_preserves_vectors_and_fingerprint():
    from mematlas.providers.local_qwen import QwenEmbeddingEngine
    engine=object.__new__(QwenEmbeddingEngine);engine.spec={"fingerprint":"a"}
    engine._exact_cache=OrderedDict();engine._exact_lock=Lock();engine.cache_hits=engine.inferred_texts=engine.inferred_tokens=engine.saved_tokens=0
    calls=[]
    def infer(texts): calls.extend(texts);return [[1.0,2.0]],3
    engine._infer=infer
    assert engine.embed(["x","x"])==([[1.0,2.0],[1.0,2.0]],6)
    assert calls==["x"] and engine.cache_hits==1
    engine.spec["fingerprint"]="b"
    assert engine.embed(["x"])==([[1.0,2.0]],3) and calls==["x","x"]


def test_generic_mem0_prefix_preserves_all_text():
    load_doctor_runtime()
    from agent.prompt import llm_messages,format_prompt,NEXT_ACTION_PROMPT
    prompt=format_prompt(NEXT_ACTION_PROMPT,{"memory_notes":["病例"*3000,"second"],"chat_history":[],"examinations":{}})
    rows=llm_messages("system",prompt,cache_prefix=True)
    assert len(rows)==3
    assert "".join(block["text"] for row in rows[1:] for block in row["content"])==prompt
    assert "second" in rows[1]["content"][0]["text"]


@pytest.mark.asyncio
async def test_empty_keyword_query_uses_native_hybrid_without_text_loss(tmp_path):
    from mematlas import Memory, MemoryEvent, MemoryUnit
    from mematlas.test.mem0.native_support import require_native, config_for, services
    require_native()
    async with Memory(config_for(tmp_path, infer=False, threshold=0), services=services()) as memory:
        await memory.add(MemoryUnit(events=[MemoryEvent("message", "clinical evidence")]))
        result = await memory.search("空关键词")
        assert result.items and result.items[0]["content"] == "clinical evidence"


@pytest.mark.asyncio
async def test_provider_probe_does_not_reserve_experiment_directory(tmp_path,monkeypatch):
    from mematlas.integrations.openhospital import suite
    from mematlas import providers
    monkeypatch.setattr(suite,"ROOT",tmp_path)
    class Client:
        async def close(self): pass
    class Embedder:
        def __init__(self,**kwargs): self._client=Client()
        async def embed(self,texts): return [[1.0]*1024]
    class LLM:
        def __init__(self,**kwargs): self.options=kwargs["default_options"];self._client=Client()
        async def complete(self,*args,**kwargs):
            from pathlib import Path
            path=Path(self.options["response_journal"]);path.parent.mkdir(parents=True,exist_ok=True);path.write_text("probe")
            return "OK"
    monkeypatch.setattr(providers,"OpenAICompatibleEmbedder",Embedder)
    monkeypatch.setattr(providers,"OpenAICompatibleLLM",LLM)
    root=tmp_path/"fresh-suite"
    config=suite.make_experiment(root,"repeated",embedding="api")
    report=await suite.check_providers(config)
    assert report["ready"] and not root.exists()
    from pathlib import Path
    assert Path(report["response_journal"]).is_file()


@pytest.mark.asyncio
async def test_native_extraction_transport_failure_does_not_commit_invalid_facts(tmp_path):
    from mematlas import Memory, MemoryEvent, MemoryUnit
    from mematlas.core.errors import MethodOperationError
    from mematlas.test.mem0.native_support import require_native, config_for, services
    require_native()
    supplied = services(facts=["clinical evidence"])
    supplied.llm.fail = True
    async with Memory(config_for(tmp_path), services=supplied) as memory:
        with pytest.raises(MethodOperationError, match="controlled provider failure"):
            await memory.add(MemoryUnit(events=[MemoryEvent("message", "clinical evidence")]))
        assert not (await memory.search("clinical evidence")).items
        supplied.llm.fail = False
        await memory.add(MemoryUnit(events=[MemoryEvent("message", "clinical evidence")]))
        assert (await memory.search("clinical evidence")).items


def test_report_deduplicates_resumed_usage_and_retains_unknown_failure(tmp_path):
    from dataclasses import asdict
    from mematlas.core.telemetry import UsageRecord
    from mematlas.integrations.openhospital.pilot_report import write_report
    job={"output":str(tmp_path/"repeated/attempt-002/evaluation")}
    (tmp_path/"suite.json").write_text(json.dumps({"pilot":True,"status":"completed","jobs":{"repeated":job}}))
    success=asdict(UsageRecord("test","test",prompt_tokens=11,completion_tokens=2,phase="doctor",stage="training"))
    failure=asdict(UsageRecord("test","test",usage_known=False,status="failed",phase="doctor",stage="training"))
    for attempt,rows in [(1,[success]),(2,[success,failure])]:
        folder=tmp_path/f"repeated/attempt-{attempt:03d}/evaluation/runs/001-mem0";folder.mkdir(parents=True)
        (folder/"usage.jsonl").write_text("\n".join(json.dumps(row) for row in rows)+"\n")
    report=write_report(tmp_path)
    row=report["usage_by_stage_phase_kind"]["training/doctor/llm"]
    assert report["unique_usage_records"]==2 and row["tracked_calls"]==2
    assert row["prompt_tokens"]==11 and row["failed_calls"]==1 and row["token_coverage"]=="partial"
