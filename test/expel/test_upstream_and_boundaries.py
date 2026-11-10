"""Pinned-source differential checks and real-storage boundary regression tests."""
from mematlas.evaluation.validation import submit_episode
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import List, Tuple, Any, Callable

import pytest

from mematlas import Memory, MemoryUnit
from mematlas.core.errors import MethodOperationError
from mematlas.methods.expel.core import ExpeLCore, parse_rules, update_rules
from mematlas.methods.expel.utils import trajectory_fields
from test_expel_integration import services, config, episode

REFERENCE = Path(__file__).parent / 'reference'


def functions(path, names, namespace=None):
    """Execute only audited pure upstream functions, avoiding its obsolete dependencies."""
    source = path.read_text(encoding='utf-8').replace('^(?i)', '(?i)^')
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = dict(re=re, List=List, Tuple=Tuple, Any=Any, Callable=Callable)
    scope.update(namespace or {})
    exec(compile(tree, str(path), 'exec'), scope)
    return scope


def test_reference_inventory():
    record = json.loads((REFERENCE/'source.json').read_text(encoding='utf-8'))
    for row in record['files']:
        raw = (REFERENCE/row['path']).read_bytes()
        assert row['sha256'] in {hashlib.sha256(raw).hexdigest(),
                                 hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()}


@pytest.mark.parametrize('full', [False, True])
def test_rule_operations_match_pinned_upstream(full):
    source = functions(REFERENCE/'agent/expel.py',
        {'parse_rules', 'update_rules', 'retrieve_rule_index', 'is_existing_rule'})
    cases = [
        'REMOVE 1: Verify evidence.\nAGREE 2: Use lookup.\nADD 3: Avoid guessing.',
        'EDIT 1: Verify the authoritative source.\nADD 5: Use lookup.',
        'EDIT 2: Verify evidence.\nREMOVE 9: Missing rule.\nADD: Missing period',
        'ADD: ADD is banned.\nADD 1: Read the source.\nAGREE 100: Use lookup.',
    ]
    for output in cases:
        rules = [('Verify evidence.', 2), ('Use lookup.', 3)]
        assert parse_rules(output) == source['parse_rules'](output)
        assert update_rules(rules, parse_rules(output), full) == source['update_rules'](
            deepcopy(rules), source['parse_rules'](output), full)


@pytest.mark.parametrize('profile,filename,text', [
    ('hotpotqa', 'hotpotQA.py', 'Thought 1: Search.\nAction 1: Search[x]\nObservation 1: x\nAction 2: Finish[x]'),
    ('alfworld', 'alfworld.py', '> think: Search\n> take apple\nYou pick up the apple.\n> put apple in bowl'),
    ('webshop', 'webshop.py', 'Action: think[Search]\nObservation: ok\n\nAction: search[book]\nObservation: found'),
])
def test_native_component_documents_match_upstream(profile, filename, text):
    scope = functions(REFERENCE/'prompts'/filename, {'CYCLER','STEP_IDENTIFIER','STEP_STRIPPER'})
    step = functions(REFERENCE/'prompts/__init__.py', {'STEP_CYCLER'})['STEP_CYCLER']
    lines = scope['CYCLER'](text)
    expected = {kind: [line for line in lines if scope['STEP_IDENTIFIER'](line) == kind]
                for kind in ['thought','action']}
    expected['step'] = step(profile, text, scope['CYCLER'], scope['STEP_IDENTIFIER'], scope['STEP_STRIPPER'])
    expected['reflection'] = ['Real prior reflection.']
    expected = {k:[v for v in vals if v.strip()] for k,vals in expected.items()}
    assert trajectory_fields(text, profile, ['Real prior reflection.']) == expected


def test_rule_full_prompt_threshold_and_zero_index_repair():
    core = ExpeLCore({})
    kwargs = dict(benchmark='hotpotqa', success_history='Actual success', fail_history=None, task=None, max_num_rules=2)
    assert 'Focus on REMOVE' not in core.critique_messages(existing_rules=['One.'], **kwargs)[-1]['content']
    assert 'Focus on REMOVE' in core.critique_messages(existing_rules=['One.', 'Two.'], **kwargs)[-1]['content']
    assert update_rules([('One.',2)], [('EDIT 0','Other.')]) == [('One.',2)]


@pytest.mark.asyncio
async def test_model_failure_rolls_back_and_retry_is_exact(tmp_path):
    supplied = services()
    real_complete = supplied.llm.complete
    async with Memory(config(tmp_path), services=supplied) as memory:
        await submit_episode(memory, episode('first', success=False))
        before = await memory._method.save()
        async def fail(*args, **kwargs):
            raise RuntimeError('controlled API failure')
        supplied.llm.complete = fail
        tail = episode('tail', end=True)
        with pytest.raises(MethodOperationError, match='controlled API failure'):
            await submit_episode(memory, tail)
        assert await memory._method.save() == before
        supplied.llm.complete = real_complete
        await submit_episode(memory, tail)
        after = await memory._method.save()
        assert len(after['successes'][episode('tail').task]) == 1
        assert len(after['critique_history']) == 2


@pytest.mark.asyncio
async def test_frozen_files_missing_component_and_strict_environment(tmp_path):
    supplied = services()
    async with Memory(config(tmp_path, fewshot_strategy='thought_similarity'), services=supplied) as memory:
        await submit_episode(memory, episode('one', end=True))
        files = {p:p.stat().st_mtime_ns for p in tmp_path.rglob('*') if p.name in {'expel.json','expel.faiss'}}
        assert files
        state = await memory._method.save()
        before = len(supplied.embedder.calls)
        assert not [i for i in (await memory.search('new')).items if i['memory_type']=='successful_trial']
        assert len(supplied.embedder.calls) == before
        used = await memory.search('new', extensions={'expel':{'thought':'Search for evidence','env_name':'other'}})
        assert not [i for i in used.items if i['memory_type']=='successful_trial']
        async with memory.read_only():
            await memory.search('new', extensions={'expel':{'thought':'Search for evidence'}})
        assert files == {p:p.stat().st_mtime_ns for p in files}
        assert state == await memory._method.save()


@pytest.mark.asyncio
async def test_distance_kind_when_thought_reranking_is_skipped(tmp_path):
    async with Memory(config(tmp_path, reranker='thought'), services=services()) as memory:
        await submit_episode(memory, episode('one', end=True))
        used = await memory.search('new task')
        trials = [i for i in used.items if i['memory_type']=='successful_trial']
        assert trials and trials[0]['metadata']['distance_kind'] == 'l2'


@pytest.mark.asyncio
async def test_invalid_embedding_is_rejected_without_partial_trial(tmp_path):
    supplied = services()
    async with Memory(config(tmp_path), services=supplied) as memory:
        receipt = await memory.add(MemoryUnit.from_episode(episode('bad')))
        before = await memory._method.save()
        async def bad(texts):
            return [[float('nan')]] * len(texts)
        supplied.embedder.embed = bad
        with pytest.raises(MethodOperationError, match='finite'):
            await memory.episode_end(receipt.episode_ref)
        assert await memory._method.save() == before


def test_config_and_rule_only_service_requirements(tmp_path):
    from mematlas.core.plugin import MethodServices
    supplied = MethodServices(llm=services().llm)
    assert Memory.check(config(tmp_path, fewshot_strategy='none'), services=supplied)['ready']
    assert not Memory.check(config(tmp_path, success_critique_num=0), services=supplied)['ready']


@pytest.mark.asyncio
@pytest.mark.parametrize('damage', ['task_order','document_task','missing_document','document_content','duplicate_document'])
async def test_corrupt_snapshot_is_rejected_before_state_or_file_changes(tmp_path, damage):
    async with Memory(config(tmp_path), services=services()) as memory:
        await submit_episode(memory, episode('one', end=True))
        before = await memory._method.save()
        damaged = deepcopy(before)
        if damage=='task_order':
            damaged['task_order']=[]
        elif damage=='document_task':
            damaged['documents'][0]['task']='NONEXISTENT_TASK'
        elif damage=='missing_document':
            damaged['documents'].pop()
        elif damage=='duplicate_document':
            damaged['documents'].append(deepcopy(damaged['documents'][0]))
        else:
            damaged['documents'][0]['content']='fabricated content'
        files = {p:p.read_bytes() for p in tmp_path.rglob('*') if p.name in {'expel.json','expel.faiss'}}
        with pytest.raises(ValueError, match='ExpeL'):
            await memory._method.load(damaged)
        assert await memory._method.save()==before
        assert files=={p:p.read_bytes() for p in files}


@pytest.mark.asyncio
async def test_native_query_normalization_and_webshop_task_document(tmp_path):
    supplied = services()
    async with Memory(config(tmp_path/'hotpot',fewshot_strategy='thought_similarity'), services=supplied) as memory:
        await submit_episode(memory, episode('one'))
        await memory.search('new',extensions={'expel':{'thought':'Thought 17: Verify record.'}})
        assert supplied.embedder.calls[-1]==['Thought: Verify record.']
    supplied = services()
    async with Memory(config(tmp_path/'shop',benchmark='webshop'),services=supplied) as memory:
        await submit_episode(memory, episode('shop',task='Instruction:\nBuy a book'))
        state=await memory._method.save()
        assert next(d for d in state['documents'] if d['type']=='task')['content']=='Instruction:\nBuy a book'
