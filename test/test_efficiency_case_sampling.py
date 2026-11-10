"""Case-level inventories remain commit-bound and never trigger extra learning."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from mematlas.core.provenance import json_sha256
from mematlas.integrations.openhospital.efficiency_metrics import RoundMeter, case_measurements, input_counts


def checkpoint(folder, position):
    directory=folder/'snapshots';directory.mkdir(parents=True,exist_ok=True)
    payload={'schema_version':6,'committed':True,'schedule_position':position}
    payload['payload_sha256']=json_sha256(payload)
    (directory/f'learned_{position}.json').write_text(json.dumps(payload),encoding='utf-8')


def test_recovery_reads_latest_only_and_rejects_corruption(tmp_path, monkeypatch):
    from mematlas.evaluation.artifacts import recovery_checkpoint
    for position in range(5):
        checkpoint(tmp_path, position)
    original = Path.read_text
    reads = []
    def read(path, *args, **kwargs):
        reads.append(path.name)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read)
    assert recovery_checkpoint(tmp_path)[0] == 4
    assert reads == ["learned_4.json"]
    path = tmp_path/'snapshots/learned_4.json'
    value = json.loads(original(path));value['committed'] = False
    path.write_text(json.dumps(value), encoding='utf8')
    with pytest.raises(ValueError, match='invalid checksum'):
        recovery_checkpoint(tmp_path)


def test_streamed_canonical_hash_matches_existing_checkpoints():
    import hashlib
    values = [None, {'中文':'a\\b"\n', 'numbers':[1, -0.0, 1e-20, True]},
              {'nested': [{'value':'医'*10000} for _ in range(20)]}, {'x':float('nan')}]
    for value in values:
        expected = hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
            separators=(',', ':'), default=str).encode('utf8')).hexdigest()
        assert json_sha256(value) == expected


@pytest.mark.asyncio
async def test_each_case_sample_flat_sizes_and_missing_point_after_resume(tmp_path,monkeypatch):
    events=[]
    class Core:
        services=SimpleNamespace(usage=SimpleNamespace(cursor=lambda:0))
        async def _freeze_backend(self):events.append('freeze')
        async def _thaw_backend(self):events.append('thaw')
    class Backend:
        async def _inspect(self,options):
            events.append('inspect')
            return {'storage_bytes':100,'memory_text_tokens':20,'background_pending':True}
    core=Core();core._method=Backend()
    engine=SimpleNamespace(memory=SimpleNamespace(core_memory=core),output_dir=tmp_path/'evaluation',
        _position=0,_segment_id='test',_curve=[])
    folder=tmp_path/'activity';folder.mkdir()
    monkeypatch.setenv('MEMATLAS_ACTIVITY_DIR',str(folder))
    monkeypatch.setenv('MEMATLAS_EFFICIENCY_OPTIONS',json.dumps({'method':'memos','tokenizer_dir':'not-used'}))
    def counts(*args,**kwargs):
        return {'rsi_tokens':engine._position*100,'payload_tokens':engine._position*200,
            'accepted_units':engine._position,'accepted_event_ledger':[]}
    monkeypatch.setattr('mematlas.integrations.openhospital.efficiency_metrics.input_counts',counts)
    meter=RoundMeter(engine);await meter.start(1)
    for position in (1,2):
        engine._position=position;checkpoint(engine.output_dir,position)
        await meter.case(position)
    meter.close()
    points=case_measurements(tmp_path,2)
    assert [p['rsi_tokens'] for p in points]==[0,100,200]
    assert [p['storage_bytes'] for p in points]==[100,100,100]
    assert all(events[i:i+3]==['freeze','inspect','thaw'] for i in range(0,len(events),3))
    assert len(case_measurements(tmp_path,1))==2
    # Checkpoint 3 was committed; sampling crashed. Resume fills only point 3.
    engine._position=3;checkpoint(engine.output_dir,3)
    resumed=RoundMeter(engine);await resumed.start(1);resumed.close()
    assert len(case_measurements(tmp_path,3))==4
    checkpoint_before=case_measurements(tmp_path,3)[-1]['payload_sha256']
    resumed=RoundMeter(engine);await resumed.start(1);resumed.close()
    assert case_measurements(tmp_path,3)[-1]['payload_sha256']==checkpoint_before
    assert events.count('inspect')==4  # Baseline, 2 cases, one missing recovered point.


@pytest.mark.asyncio
async def test_token_cache_counts_new_units_only_but_still_keeps_retry_transfer(tmp_path,monkeypatch):
    from mematlas import Memory,MemoryEvent,MemoryUnit,PluginRegistry
    from mematlas.test.test_memory_interface import RecordingPlugin
    registry=PluginRegistry();registry.register(RecordingPlugin())
    calls=[]
    def counter(text,directory):calls.append(text);return len(text)
    monkeypatch.setattr('mematlas.integrations.openhospital.efficiency_metrics.count',counter)
    cfg={'method':{'method_id':'recording'},'scope':{'user_id':'test'},'runtime':{'artifact_dir':str(tmp_path/'m')}}
    async with Memory(cfg,registry=registry) as memory:
        unit=MemoryUnit(events=[MemoryEvent('message','one',event_id='one')],write_ref='one',episode_ref='case')
        await memory.add(unit);cache={}
        first=input_counts(memory,'unused',tmp_path,cache=cache);number=len(calls)
        await memory.add(unit)
        same=input_counts(memory,'unused',tmp_path,cache=cache)
        assert same['rsi_tokens']==first['rsi_tokens'] and len(calls)==number
        await memory.add(MemoryUnit(events=[MemoryEvent('message','two',event_id='two')],write_ref='two',episode_ref='next'))
        second=input_counts(memory,'unused',tmp_path,cache=cache)
        assert second['rsi_tokens']==6 and len(calls)==number+2


def test_corrupted_case_point_fails_closed(tmp_path):
    path=tmp_path/'activity/cases';path.mkdir(parents=True)
    (path/'case-0001.json').write_text(json.dumps({'schema':1,'case_position':1,'rsi_tokens':99}),encoding='utf-8')
    with pytest.raises(ValueError,match='checksum'):
        case_measurements(tmp_path,1)
