from mematlas.evaluation.validation import submit_episode
from mematlas.test.evaluation.support import add_event
from copy import deepcopy
from dataclasses import replace
import json
import pytest
from mematlas import Memory,MemAtlasConfig,MemoryUnit
from mematlas.core.errors import MethodOperationError
from awm_fixtures import config,services,episode,WORKFLOW


def test_user_entry_v2_preflight_has_no_allocation(tmp_path):
    from mematlas.methods.awm import AWMConfig,SUPPORTED_OPERATIONS,AddOptions,SearchOptions
    value=MemAtlasConfig(method=AWMConfig(profile='mind2web'))
    assert value.method.method_id=='awm'
    assert set(SUPPORTED_OPERATIONS)=={'index','add','search','episode_end','save','load','reset'}
    assert Memory.check(config(tmp_path),services=services())['ready']
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_online_learns_then_retrieve_is_pure_and_scoped(tmp_path):
    supplied=services()
    async with Memory(config(tmp_path),services=supplied) as memory:
        e=episode('one')
        for event in e.events:await add_event(memory, replace(event,episode_id=e.episode_id))
        await submit_episode(memory, replace(e,events=None))
        await submit_episode(memory, replace(e,events=None))
        assert len(supplied.llm.calls)==1
        before=await memory._method.save()
        files={p:p.stat().st_mtime_ns for p in tmp_path.rglob('awm.json')}
        async with memory.read_only():
            used=await memory.search('any query',extensions={'awm':{'website':'shop'}})
        assert used.items[0]['content']==WORKFLOW
        assert set(used.items[0]['source_event_ids'])=={'one-obs','one-action'}
        assert before==await memory._method.save() and len(supplied.llm.calls)==1
        assert files=={p:p.stat().st_mtime_ns for p in files}
        assert not (await memory.search('q',extensions={'awm':{'website':'elsewhere'}})).items


@pytest.mark.asyncio
async def test_offline_finite_tail_snapshot_and_reset(tmp_path):
    supplied=services();path=tmp_path/'saved.json'
    async with Memory(config(tmp_path/'a',induction_mode='offline'),services=supplied) as memory:
        await submit_episode(memory, episode('one',canonical=True))
        await memory.save(path)
        assert not supplied.llm.calls
        assert not (await memory.search('q',extensions={'awm':{'website':'shop'}})).items
    async with Memory(config(tmp_path/'b',induction_mode='offline'),services=supplied) as memory:
        await memory.load(path)
        await submit_episode(memory, episode('two',website='second',canonical=True,end=True))
        assert len(supplied.llm.calls)==2
        assert supplied.llm.calls[0]['options']['max_tokens']==1024
        await memory.reset()
        assert not (await memory.search('q',extensions={'awm':{'website':'shop'}})).items


@pytest.mark.asyncio
async def test_webarena_success_filter_and_template_sampling(tmp_path):
    supplied=services()
    async with Memory(config(tmp_path,profile='webarena'),services=supplied) as memory:
        await submit_episode(memory, episode('failed',success=False))
        assert not supplied.llm.calls
        await submit_episode(memory, episode('one',template_id=1))
        await submit_episode(memory, episode('two',template_id=1))
        await submit_episode(memory, episode('three',template_id=2))
        state=await memory._method.save()
        latest=state['induction_history'][-1]
        assert len(latest['selected_episode_ids'])==2
        assert 'failed' not in latest['selected_episode_ids']
        assert supplied.llm.calls[-1]['options']=={'temperature':1.0,'max_tokens':2048}


@pytest.mark.asyncio
async def test_failed_induction_rolls_back_and_retry_is_exact(tmp_path):
    supplied=services()
    async with Memory(config(tmp_path),services=supplied) as memory:
        value=episode('one')
        receipt=await memory.add(MemoryUnit.from_episode(value))
        before=await memory._method.save()
        real=supplied.llm.complete
        async def bad(*args,**kwargs):raise RuntimeError('controlled failure')
        supplied.llm.complete=bad
        with pytest.raises(MethodOperationError,match='controlled failure'):await memory.episode_end(receipt.episode_ref)
        assert await memory._method.save()==before
        supplied.llm.complete=real
        await memory.episode_end(receipt.episode_ref)
        assert len((await memory._method.save())['induction_history'])==1


@pytest.mark.asyncio
async def test_offline_canonical_and_missing_website_are_not_invented(tmp_path):
    supplied=services()
    async with Memory(config(tmp_path,induction_mode='offline'),services=supplied) as memory:
        receipt=await memory.add(MemoryUnit.from_episode(episode('one')))
        before=await memory._method.save()
        with pytest.raises(MethodOperationError,match='canonical'):await memory.episode_end(receipt.episode_ref)
        assert before==await memory._method.save() and not supplied.llm.calls
        with pytest.raises(Exception,match='website'):await memory.search('q',extensions={'awm':{}})


@pytest.mark.asyncio
@pytest.mark.parametrize('damage',['task_source','missing_workflow','scope','pending'])
async def test_corrupt_snapshot_rejected_without_mutation(tmp_path,damage):
    supplied=services()
    async with Memory(config(tmp_path),services=supplied) as memory:
        await submit_episode(memory, episode('one'))
        before=await memory._method.save();bad=deepcopy(before)
        if damage=='task_source':bad['induction_history'][0]['selected_episode_ids']=['absent']
        elif damage=='missing_workflow':bad['workflow_text']={}
        elif damage=='scope':bad['active_scope']=None
        else:bad['pending']={'absent':1}
        with pytest.raises(ValueError,match='AWM'):await memory._method.load(bad)
        assert before==await memory._method.save()


@pytest.mark.asyncio
async def test_failed_setup_never_overwrites_invalid_local_state(tmp_path,monkeypatch):
    from mematlas.methods.awm import method
    path=tmp_path/'awm.json';original=b'{"version":1,"important_old_state":"keep for migration"}'
    path.write_bytes(original)
    monkeypatch.setattr(method,'configured_file',lambda *args,**kwargs:path)
    with pytest.raises(MethodOperationError,match='AWM'):
        async with Memory(config(tmp_path/'runtime'),services=services()):pass
    assert path.read_bytes()==original
