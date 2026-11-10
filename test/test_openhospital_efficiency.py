"""Controlled invariants for durable native efficiency measurement."""
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
from io import BytesIO
from urllib.error import HTTPError
import pytest

from mematlas.core.activity import summarize, union_seconds
from mematlas.core.recovery import error_info
from mematlas.integrations.openhospital.efficiency import SuiteLock, make_config, METHODS
from mematlas.integrations.openhospital.efficiency_metrics import input_counts
from mematlas.integrations.openhospital.inspection import files_inventory


def write_events(path, events):
    path.mkdir(parents=True, exist_ok=True)
    (path / "events-1.jsonl").write_text("\n".join(json.dumps(r) for r in events), encoding="utf-8")


def test_overlap_nested_background_rounds_and_segments(tmp_path):
    events = []
    for key, category, start, end, round_ in [("a","foreground",1,5,1),("b","foreground",2,3,1),
            ("c","background",4,8,1),("d","persistence",3,6,1),("e","foreground",20,22,2),
            ("f","foreground",100,102,1)]:
        events.extend([{"kind":"start","span_id":key,"category":category,"clock_ns":int(start*1e9),"round":round_,"pid":1},
                       {"kind":"end","span_id":key,"clock_ns":int(end*1e9),"pid":1}])
    write_events(tmp_path,events)
    result = summarize(tmp_path,1)
    assert result["memory_method_wall_s"] == 9
    assert result["memory_foreground_wait_s"] == 6
    assert result["memory_background_active_s"] == 4
    assert result["memory_persistence_s"] == 3
    assert summarize(tmp_path,2)["memory_method_wall_s"] == 2


def test_killed_span_reports_lower_bound_and_unknown_tail(tmp_path):
    write_events(tmp_path,[{"kind":"start","span_id":"a","category":"foreground","clock_ns":10,"pid":1,"round":1},
                           {"kind":"heartbeat","pid":1,"clock_ns":100}])
    result=summarize(tmp_path,1)
    assert not result["timing_complete"] and result["unobserved_tail_bound_s"] is None
    assert result["unfinished_spans"] == ["a"]


def test_os_lock_excludes_second_owner_and_releases(tmp_path):
    with SuiteLock(tmp_path):
        with pytest.raises(RuntimeError,match="active owner"):
            with SuiteLock(tmp_path):
                pytest.fail("duplicate owner")
    with SuiteLock(tmp_path):
        pass


@pytest.mark.parametrize("status,expected",[(429,True),(503,True),(401,False),(403,False),(404,False)])
def test_cause_chain_and_remote_classification(status, expected):
    native=RuntimeError("provider")
    native.status_code=status
    wrapped=RuntimeError("executor")
    wrapped.remote_error=error_info(native)
    outer=RuntimeError("public memory")
    outer.__cause__=wrapped
    assert error_info(outer)["recoverable"] is expected


@pytest.mark.asyncio
async def test_embedding_exhaustion_preserves_eight_attempts_and_retry_after(monkeypatch):
    from mematlas.providers import requests
    from mematlas.core.telemetry import UsageLedger
    delays=[]
    async def sleep(delay): delays.append(delay)
    monkeypatch.setattr(requests.asyncio,"sleep",sleep)
    async def request():
        exc=RuntimeError("rate limited")
        exc.status_code=429
        exc.response=SimpleNamespace(headers={"Retry-After":"40"})
        raise exc
    ledger=UsageLedger()
    with pytest.raises(requests.LLMRequestPaused):
        await requests.metered_request(request,ledger=ledger,provider="test",model="e",kind="embedding")
    assert delays == [40]*7 and ledger.summary()["failed_calls"] == 8
    records=ledger.to_dicts()
    assert len({r["metadata"]["request_id"] for r in records}) == 1
    assert all(not r["usage_known"] for r in records)


def test_storage_includes_external_transactions_excludes_snapshots(tmp_path):
    root=tmp_path / "live"
    for name in ("graph/index.db","transactions/transaction.0","db.sqlite-wal","snapshots/copy.db","logs/events.log"):
        p=root / name
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(b"12345")
    result=files_inventory([root])
    assert len(result)==3 and sum(r["bytes"] for r in result.values())==15
    assert sum(r["bytes"] for r in result.values() if r["category"]=="wal_transactions")==10


@pytest.mark.asyncio
async def test_accepted_event_identity_and_resume_dedup(tmp_path):
    from mematlas import Memory,MemoryEvent,MemoryUnit,PluginRegistry
    from mematlas.test.test_memory_interface import RecordingPlugin
    from mematlas.integrations.openhospital.inspection import count
    registry=PluginRegistry();registry.register(RecordingPlugin())
    model_dir=Path(__file__).resolve().parents[1] / "models/Qwen3-Embedding-0.6B"
    event=MemoryEvent("message","a long clinical observation "*100, event_id="same-event")
    unit=MemoryUnit(events=[event], write_ref="one", episode_ref="case")
    config={"method":{"method_id":"recording"},"scope":{"user_id":"test"},"runtime":{"artifact_dir":str(tmp_path/"m")}}
    async with Memory(config,registry=registry) as memory:
        await memory.add(unit)
        await memory.add(unit)
        counts=input_counts(memory,str(model_dir),tmp_path)
        assert len(counts["accepted_event_ledger"])==1
        assert counts["rsi_tokens"]==count(event.content,model_dir)
        await memory.save(tmp_path/"checkpoint.json")
    config["runtime"]["artifact_dir"]=str(tmp_path/"restored")
    async with Memory(config,registry=registry) as memory:
        await memory.load(tmp_path/"checkpoint.json")
        assert input_counts(memory,str(model_dir),tmp_path)["rsi_tokens"]==counts["rsi_tokens"]


def test_formal_and_smoke_schedule_identity_isolation(tmp_path):
    from mematlas.integrations.openhospital.planning import build_plan
    from mematlas.integrations.openhospital.config import HospitalOptions
    from mematlas.integrations.openhospital.efficiency import ROOT
    settings={"identity":"controlled", "smoke":False,"bolt_port":18687,"hospital_port":18001,"embedding_port":18766,
              "model_dir":str(ROOT/"models/Qwen3-Embedding-0.6B")}
    configs=[make_config(tmp_path,m,settings) for m in METHODS]
    plans=[build_plan(HospitalOptions.from_experiment(c),checkpoints=(),seed=0).plan for c in configs]
    assert all(p.train_ids==plans[0].train_ids for p in plans)
    assert len(plans[0].train_ids)==100 and len(set(plans[0].train_ids))==100
    assert [len(b) for b in plans[0].batches()]==[20]*5
    assert len({c.methods[0].scope["user_id"] for c in configs})==5
    assert len({c.benchmark["world_dir"] for c in configs})==5
    assert all(c.benchmark["memory_top_k"] is None for c in configs)
    settings["smoke"]=True
    plan=build_plan(HospitalOptions.from_experiment(make_config(tmp_path,"memos",settings)),checkpoints=(),seed=0).plan
    assert plan.train_ids==plans[0].train_ids[:2] and [len(b) for b in plan.batches()]==[1,1]


@pytest.mark.asyncio
async def test_json_fallback_salvages_valid_items_and_filters_duplicates(tmp_path):
    from mematlas.integrations.openhospital.application import load_doctor_runtime
    cls,_=load_doctor_runtime()
    doctor=cls(config={"json_exam_fallback":True,"rsi_strict_outputs":True})
    doctor._timeline=[]
    async def llm(*args,**kwargs):
        return '{"examinations":["CBC","CBC","invalid","done"],"reason":1}'
    doctor._llm_text=llm
    result=await doctor._call_llm(prompt="test",default={"examinations":["other"],"reason":"fallback"},
        prompt_name="exam_item",allowed_examinations=["CBC","done","other"],already_ordered=["done"])
    assert result["examinations"]==["CBC"]
    assert doctor._timeline[-1]["json_fallback"] and doctor._timeline[-1]["json_repair_count"]==3


def test_hospital_request_response_loss_does_not_repeat_action(tmp_path):
    from mematlas.integrations.openhospital.application import load_doctor_runtime
    load_doctor_runtime()
    from mematlas.integrations.openhospital_runtime.local_world import HospitalWorld
    world=HospitalWorld(tmp_path,{},dict)
    payload={"team_id":"one","_request_id":"same","items":["CBC"]}
    assert world.cached_response("one",payload) is None
    world.cache_response("one",payload,{"result":"original"})
    assert world.cached_response("one",payload)=={"result":"original"}
    with pytest.raises(ValueError,match="different payload"):
        world.cached_response("one",{**payload,"items":["other"]})


def test_attempt_restart_keeps_semantic_config_and_diagnostic_identity(tmp_path, monkeypatch):
    from mematlas.integrations.openhospital import efficiency
    settings={"identity":"controlled", "smoke":True,"bolt_port":18687,"hospital_port":18001,"embedding_port":18766,
              "model_dir":str(efficiency.ROOT/"models/Qwen3-Embedding-0.6B")}
    config=make_config(tmp_path,"memos",settings)
    path=tmp_path/"memos/attempt-001/config.json"
    from mematlas.core.files import atomic_write_json
    atomic_write_json(path,config.to_dict())
    manifest={"settings":settings,"jobs":{"memos":{"attempt":1,"config":str(path),"output":config.output_dir,"source":None}}}
    monkeypatch.setattr(efficiency,"recovery_position",lambda *args:(1,config.output_dir))
    efficiency.next_attempt(tmp_path,manifest,"memos")
    recovered=efficiency.ExperimentConfig.load(manifest["jobs"]["memos"]["config"])
    assert recovered.methods[0].providers==config.methods[0].providers
    assert recovered.methods[0].scope==config.methods[0].scope
    assert recovered.methods[0].method==config.methods[0].method
    assert recovered.benchmark==config.benchmark
    assert recovered.methods[0].runtime.artifact_dir!=config.methods[0].runtime.artifact_dir


@pytest.mark.asyncio
async def test_real_worker_keeps_recoverability_and_cause_chain(tmp_path, monkeypatch):
    from mematlas import Memory,PluginRegistry
    from mematlas.core.errors import MethodOperationError
    from mematlas.test.native_adapters.fixtures import RecoverableFailurePlugin
    registry=PluginRegistry();registry.register(RecoverableFailurePlugin())
    monkeypatch.setenv("MEMATLAS_ACTIVITY_DIR",str(tmp_path/"activity"))
    async with Memory({"method":{"method_id":"recording"},"scope":{"user_id":"test"},
        "runtime":{"executor":"subprocess","artifact_dir":str(tmp_path/"memory")}},registry=registry) as memory:
        with pytest.raises(MethodOperationError) as caught:
            await memory.search("trigger controlled timeout")
        error=error_info(caught.value)
        assert error["recoverable"]
        assert {row["type"] for row in error["chain"]}>={"MethodOperationError","ExecutorError","LLMRequestPaused","TimeoutError"}
    timing=summarize(tmp_path/"activity")
    assert timing["memory_foreground_wait_s"]>0 and timing["timing_complete"]


@pytest.mark.asyncio
async def test_local_embedding_transport_batch_preserves_order_and_actual_calls():
    from mematlas.providers.embeddings import OpenAICompatibleEmbedder
    from mematlas.core.telemetry import UsageLedger
    calls=[]
    async def create(**kwargs):
        calls.append(kwargs["input"])
        return SimpleNamespace(data=[SimpleNamespace(embedding=[float(t)]) for t in kwargs["input"]],
            usage=SimpleNamespace(prompt_tokens=len(kwargs["input"]),completion_tokens=0))
    provider=object.__new__(OpenAICompatibleEmbedder)
    provider.model="Qwen/Qwen3-Embedding-0.6B"
    provider.base_url="http://127.0.0.1:18766/v1"
    provider.dimensions=1024
    provider.usage=UsageLedger()
    provider._client=SimpleNamespace(embeddings=SimpleNamespace(create=create))
    assert await provider.embed([str(i) for i in range(80)])==[[float(i)] for i in range(80)]
    assert [len(batch) for batch in calls]==[32,32,16]
    assert provider.usage.summary()["embedding_calls"]==3


def test_exception_group_preserves_all_native_causes_and_permanent_errors():
    temporary = TimeoutError("background provider timeout")
    grouped = ExceptionGroup("native tasks failed", [temporary, ConnectionError("database connection")])
    info = error_info(grouped)
    assert info["recoverable"] and {r["type"] for r in info["chain"]} >= {"TimeoutError", "ConnectionError"}
    permanent = RuntimeError("invalid structured output configuration")
    permanent.status_code = 400
    assert not error_info(ExceptionGroup("mixed failure", [grouped, permanent]))["recoverable"]


@pytest.mark.asyncio
async def test_auto_recovery_wait_counted_as_total_without_memory_inflation(tmp_path, monkeypatch):
    from mematlas.integrations.openhospital import efficiency
    import time
    folder = tmp_path / "mem0/activity"
    folder.mkdir(parents=True)
    (folder / "context.json").write_text(json.dumps({"round":2,"method":"mem0"}),encoding="utf-8")
    ticks = iter([0, 0, 31])
    monkeypatch.setattr(efficiency.time,"time",lambda:next(ticks))
    monkeypatch.setattr(efficiency,"terminal_status",lambda *a:None)
    async def sleep(_): pass
    monkeypatch.setattr(efficiency.asyncio,"sleep",sleep)
    await efficiency.recovery_wait(tmp_path,"mem0",{"retry_at":30},{},tmp_path/"STOP")
    result=summarize(folder,2)
    assert result["round_total_s"]>0 and result["automatic_recovery_wait_s"]>0
    assert result["memory_method_wall_s"]==0 and result["timing_complete"]


@pytest.mark.asyncio
async def test_worker_keeps_failed_attempt_costs_even_without_its_snapshot(tmp_path,monkeypatch):
    from mematlas.integrations.openhospital import efficiency
    from mematlas.core.telemetry import UsageLedger,UsageRecord
    shared = UsageRecord("fixture","model",prompt_tokens=10,completion_tokens=2,record_id="shared")
    for attempt in (1,2):
        ledger=UsageLedger()
        ledger.add(shared,imported=True)
        if attempt==2:
            ledger.add(UsageRecord("fixture","model",usage_known=False,status="failed",record_id="failed-new"),imported=True)
        ledger.bind_journal(tmp_path/f"mem0/attempt-{attempt:03d}/evaluation/runs/001-mem0/usage.jsonl")
    config=SimpleNamespace()
    monkeypatch.setattr(efficiency.ExperimentConfig,"load",lambda _:config)
    monkeypatch.setattr(efficiency,"embedding_ledger",lambda _:UsageLedger())
    captured=[]
    async def evaluate(*args,services,**kwargs):
        captured.extend(services.usage.to_dicts())
        return [SimpleNamespace(failures=[])]
    monkeypatch.setattr(efficiency.ExperimentRunner,"evaluate",evaluate)
    directory=tmp_path/"mem0/attempt-003";directory.mkdir(parents=True)
    assert await efficiency.worker(SimpleNamespace(config=str(directory/"config.json"),source="old-valid-snapshot"))==0
    assert {r["record_id"] for r in captured}=={"shared","failed-new"}


def test_round_requests_include_previous_attempts_and_native_background(tmp_path):
    from mematlas.core.telemetry import UsageLedger,UsageRecord
    from mematlas.integrations.openhospital.efficiency_metrics import request_inventory,request_summary
    first=UsageRecord('provider','model',phase='doctor',prompt_tokens=10,completion_tokens=2,
        record_id='first',case_id='batch:1:patient:P1',timestamp='2026-11-06T01:00:01+00:00')
    failed=UsageRecord('provider','model',phase='memory',usage_known=False,status='failed',
        record_id='failed',case_id='batch:1:patient:P1',timestamp='2026-11-06T01:00:02+00:00')
    background=UsageRecord('provider','model',phase='memory',prompt_tokens=20,completion_tokens=1,
        record_id='background',timestamp='2026-11-06T01:01:01+00:00')
    for attempt,records in [(1,[first,failed]),(2,[first,failed,background])]:
        ledger=UsageLedger()
        for row in records:ledger.add(row,imported=True)
        ledger.bind_journal(tmp_path/f'attempt-{attempt:03d}/evaluation/runs/001-method/usage.jsonl')
    from datetime import datetime
    start=int(datetime.fromisoformat('2026-11-06T01:01:00+00:00').timestamp()*1e9)
    write_events(tmp_path/'activity',[
        {'kind':'start','span_id':'native-batch','category':'background','clock_ns':1,'utc_ns':start,'round':2},
        {'kind':'end','span_id':'native-batch','clock_ns':2,'utc_ns':start+int(2e9)}])
    assigned,unassigned=request_inventory(tmp_path)
    assert not unassigned and [len(assigned[i]) for i in (1,2)]==[2,1]
    summary=request_summary(assigned[1])
    assert summary['actual_doctor_calls']==1 and summary['actual_memory_llm_failed_calls']==1
    assert summary['actual_memory_llm_tokens'] is None and summary['actual_unknown_usage_requests']==1


def test_atomic_json_waits_out_windows_reader_without_deleting_old_target(tmp_path, monkeypatch):
    from mematlas.core import files
    import time
    target=tmp_path/'value.json'
    files.atomic_write_json(target,{'old':True})
    real=files.os.replace; attempts=[]; delays=[]
    def replace(source,dest):
        attempts.append(1)
        if len(attempts)<10:
            assert json.loads(target.read_text())=={'old':True}
            exc=PermissionError('reader busy'); exc.winerror=5; raise exc
        return real(source,dest)
    monkeypatch.setattr(files.os,'replace',replace)
    monkeypatch.setattr(time,'sleep',lambda delay:delays.append(delay))
    files.atomic_write_json(target,{'new':True})
    assert json.loads(target.read_text())=={'new':True} and len(attempts)==10
    assert sum(delays)>4 and max(delays)<=1


def test_activity_chunking_incremental_partial_unicode_and_truncation(tmp_path, monkeypatch):
    from mematlas.core import activity as journal
    monkeypatch.setattr(journal, '_READ_CHUNK_BYTES', 37)
    path = tmp_path / 'events-1.jsonl'
    payload = {'kind':'submission', 'text':'长轨迹🙂'*1000}
    start = {'kind':'start', 'span_id':'one', 'category':'foreground', 'clock_ns':0, 'pid':1}
    end = {'kind':'end', 'span_id':'one', 'clock_ns':int(2e9)}
    raw = (json.dumps(payload,ensure_ascii=False)+'\n'+json.dumps(start)+'\n').encode()
    path.write_bytes(raw)
    original = journal._chunk
    positions = []
    def observed(stream):
        positions.append(stream.tell())
        return original(stream)
    monkeypatch.setattr(journal, '_chunk', observed)
    assert not journal.summarize(tmp_path)['timing_complete']
    assert journal.rows(tmp_path)[0] == payload
    positions.clear()
    journal.summarize(tmp_path)
    assert positions == [len(raw)]  # Cached records are not reread.
    encoded = json.dumps(end).encode()
    with path.open('ab') as stream: stream.write(encoded[:20])
    assert not journal.summarize(tmp_path)['timing_complete']
    with path.open('ab') as stream: stream.write(encoded[20:]+b'\n')
    assert journal.summarize(tmp_path)['memory_method_wall_s'] == 2
    path.write_text(json.dumps({'kind':'heartbeat','pid':1,'clock_ns':0})+'\n')
    assert journal.summarize(tmp_path)['memory_method_wall_s'] == 0


def test_activity_einval_bounded_retry(monkeypatch):
    from mematlas.core import activity as journal
    import errno
    calls = []
    class Stream:
        def read(self, size):
            calls.append(size)
            if len(calls) < 4: raise OSError(errno.EINVAL, 'transient')
            return b'ok'
    monkeypatch.setattr(journal.time, 'sleep', lambda _:None)
    assert journal._chunk(Stream()) == b'ok'
    assert calls == [journal._READ_CHUNK_BYTES]*4
    class Failed:
        def read(self, size): raise OSError(errno.EINVAL, 'persistent')
    with pytest.raises(OSError): journal._chunk(Failed())


@pytest.mark.parametrize('failure_at', ['collect','console'])
def test_progress_observer_cannot_terminate_worker(tmp_path, monkeypatch, failure_at):
    from mematlas.integrations.openhospital import efficiency, dashboard
    def fail(*args): raise OSError(22, 'Invalid argument')
    monkeypatch.setattr(dashboard,'collect',fail if failure_at=='collect' else lambda *a:{})
    monkeypatch.setattr(dashboard,'render',lambda _: 'screen')
    if failure_at=='console': monkeypatch.setattr(dashboard.Console,'show',fail)
    efficiency.terminal_status(tmp_path)
    warning=json.loads((tmp_path/'observer-warning.json').read_text())
    assert warning['clinical_worker_continues'] and warning['type']=='OSError'


def test_completed_cleanup_preserves_results_and_paused_recovery(tmp_path):
    from mematlas.integrations.openhospital import efficiency
    from mematlas.core.files import atomic_write_json
    from mematlas.core.provenance import json_sha256
    output=tmp_path/'mem0/attempt-001/evaluation'
    root=output/'runs/001-mem0'
    manifest={'settings':{'smoke':True},'jobs':{'mem0':{'status':'paused','position':2,'output':str(output)}}}
    atomic_write_json(output/'results.json',[{'method_id':'mem0','failures':[],'predictions':[{},{}]}])
    atomic_write_json(root/'curve.json',[{'batch_index':1},{'batch_index':2}])
    for index in range(3):
        point={'schema':1,'case_position':index}
        atomic_write_json(tmp_path/f'mem0/activity/cases/case-{index:04d}.json',
                          {**point,'payload_sha256':json_sha256(point)})
        atomic_write_json(tmp_path/f'mem0/activity/measurement-{index}.json',{'round':index})
    snapshot=root/'snapshots/learned_2.json'
    atomic_write_json(snapshot,{'recovery':'only'})
    native=tmp_path/'mem0/attempt-001/memory/runs/id/snapshots/mem0/one.zlib'
    native.parent.mkdir(parents=True);native.write_bytes(b'recovery copy')
    live=native.parents[2]/'mem0-native/data.db';live.parent.mkdir();live.write_bytes(b'actual memory')
    atomic_write_json(root/'usage.json',{'tokens':12})
    with pytest.raises(ValueError,match='completed'):efficiency.cleanup_completed_checkpoints(tmp_path,manifest,'mem0')
    assert snapshot.exists()
    manifest['jobs']['mem0']['status']='completed'
    # Missing statistics must prevent all deletion.
    (tmp_path/'mem0/activity/measurement-2.json').unlink()
    with pytest.raises(ValueError,match='round measurements'):efficiency.cleanup_completed_checkpoints(tmp_path,manifest,'mem0')
    assert snapshot.exists() and native.exists()
    atomic_write_json(tmp_path/'mem0/activity/measurement-2.json',{'round':2})
    receipt=efficiency.cleanup_completed_checkpoints(tmp_path,manifest,'mem0')
    assert receipt['status']=='completed' and receipt['deleted_bytes']>0
    assert not snapshot.exists() and not native.exists()
    assert live.read_bytes()==b'actual memory' and (root/'usage.json').exists() and (output/'results.json').exists()
    assert efficiency.cleanup_completed_checkpoints(tmp_path,manifest,'mem0')==receipt


def test_recovery_hotfix_requires_audited_hashes_and_identical_clinical_contract(tmp_path):
    from mematlas.integrations.openhospital.application import audited_source_repair
    from mematlas.core.files import atomic_write_json
    old={'doctor':'unchanged','framework_sources':{'core/activity.py':'old','native.py':'same'}}
    new={**old,'framework_sources':{'core/activity.py':'new','native.py':'same'}}
    assert not audited_source_repair(old,new,tmp_path)
    audit=tmp_path/'repairs/one.json'
    atomic_write_json(audit,{'changes':{'core/activity.py':{'before':'old','after':'new'}}})
    atomic_write_json(tmp_path/'suite.json',{'source_repairs':[str(audit)],'source_signature':new['framework_sources']})
    assert audited_source_repair(old,new,tmp_path)
    assert not audited_source_repair(old,{**new,'doctor':'different'},tmp_path)
    assert not audited_source_repair(old,{**new,'framework_sources':{**new['framework_sources'],'native.py':'different'}},tmp_path)
    atomic_write_json(audit,{'changes':{'core/activity.py':{'before':'wrong','after':'new'}}})
    assert not audited_source_repair(old,new,tmp_path)


@pytest.mark.asyncio
async def test_missing_diagnosis_reasoning_preserves_actual_clinical_output_and_costs():
    from mematlas.integrations.openhospital.application import load_doctor_runtime
    cls,_=load_doctor_runtime()
    doctor=cls(config={'json_exam_fallback':True,'rsi_strict_outputs':True}); doctor._timeline=[]
    original={'diagnosis':'actual chosen disease','treatment_plan':'exact original treatment'}
    responses=[json.dumps(original)]*4; calls=[]
    async def llm(*a,**kw): calls.append(kw); return responses.pop(0)
    doctor._llm_text=llm
    default={'diagnosis':'DO NOT SUBSTITUTE','treatment_plan':'DO NOT SUBSTITUTE','reasoning':'DO NOT INVENT'}
    result=await doctor._call_llm(prompt='original clinical input',default=default,prompt_name='disease_and_treatment')
    assert result=={**original,'reasoning':''} and len(calls)==4
    assert doctor._timeline[-1]['json_repair_count']==3 and doctor._timeline[-1]['json_fallback']


@pytest.mark.asyncio
async def test_absent_treatment_still_pauses_recoverably_without_fake_default():
    from mematlas.integrations.openhospital.application import load_doctor_runtime
    cls,_=load_doctor_runtime()
    doctor=cls(config={'json_exam_fallback':True,'rsi_strict_outputs':True});doctor._timeline=[]
    async def llm(*a,**kw): return '{"diagnosis":"chosen"}'
    doctor._llm_text=llm
    with pytest.raises(RuntimeError) as caught:
        await doctor._call_llm(prompt='input',default={'diagnosis':'DO NOT USE','treatment_plan':'DO NOT USE','reasoning':''},prompt_name='disease_and_treatment')
    assert error_info(caught.value)['recoverable'] and doctor._timeline==[]


def test_audited_repairs_support_method_started_after_prior_hotfix(tmp_path):
    from mematlas.integrations.openhospital.application import audited_source_repair
    from mematlas.core.files import atomic_write_json
    key='integrations/openhospital_runtime/hospital_agent_example/agent/agent.py'
    prior={'doctor':'same','framework_sources':{'one.py':'middle'},'scene_sources':{'hospital_agent_example/agent/agent.py':'old'}}
    current={**prior,'framework_sources':{'one.py':'new'},'scene_sources':{'hospital_agent_example/agent/agent.py':'new-scene'}}
    a=tmp_path/'repairs/one.json';b=tmp_path/'repairs/two.json'
    atomic_write_json(a,{'changes':{'one.py':{'before':'old','after':'middle'}}})
    atomic_write_json(b,{'changes':{'one.py':{'before':'middle','after':'new'},key:{'before':'old','after':'new-scene'}}})
    atomic_write_json(tmp_path/'suite.json',{'source_repairs':[str(a),str(b)],'source_signature':{'one.py':'new',key:'new-scene'}})
    assert audited_source_repair(prior,current,tmp_path)
    assert not audited_source_repair(prior,{**current,'doctor':'changed'},tmp_path)
