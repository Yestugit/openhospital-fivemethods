"""Real FAISS storage with explicit deterministic model fixtures, never quality claims."""
from mematlas.evaluation.validation import submit_episode
from mematlas.test.evaluation.support import add_event
from copy import deepcopy
import json
from pathlib import Path

import pytest

from mematlas import Memory, MemoryEpisode, MemoryEvent
from mematlas.core.plugin import MethodServices


class Rules:
    def __init__(self):
        self.calls = []

    async def complete(self, messages, **options):
        self.calls.append(deepcopy(messages))
        return 'ADD: Verify the observation before choosing the final answer.'


class Vectors:
    def __init__(self):
        self.calls = []

    async def embed(self, texts):
        self.calls.append(list(texts))
        return [[10., 0., 0.] if 'distant' in text else [1., 1., 0.] for text in texts]


def services():
    return MethodServices(llm=Rules(), embedder=Vectors())


def config(path, **params):
    return {'method': {'method_id': 'expel', 'params': {'benchmark': 'hotpotqa', **params}},
            'scope': {'user_id': 'alice'}, 'runtime': {'artifact_dir': str(path)}}


def episode(identity, task='Find the verified value', *, success=True, end=False, text=None):
    trajectory = text or ('Thought 1: Check the source.\nAction 1: Lookup[value]\n'
                          'Observation 1: The verified value is 7.\nAction 2: Finish[7]')
    return MemoryEpisode(identity, task=task, success=success,
        events=[MemoryEvent('observation', trajectory, event_id=identity+'-event')],
        extensions={'expel': {'end_of_training': end}})


def test_v2_user_entry_and_preflight(tmp_path):
    from mematlas.methods.expel import ExpeLConfig, SUPPORTED_OPERATIONS
    from mematlas import MemAtlasConfig
    config_value = MemAtlasConfig(method=ExpeLConfig(benchmark='hotpotqa'))
    assert config_value.method.method_id == 'expel'
    assert set(SUPPORTED_OPERATIONS) == {'index', 'add', 'search', 'episode_end', 'reset', 'save', 'load'}
    assert Memory.check(config(tmp_path), services=services())['ready']
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_finite_batch_compares_pairs_then_successes_and_never_learns_in_retrieve(tmp_path):
    supplied = services()
    async with Memory(config(tmp_path), services=supplied) as memory:
        await submit_episode(memory, episode('failure', success=False))
        await submit_episode(memory, episode('success'))
        assert supplied.llm.calls == []
        before = await memory._method.save()
        await memory.search('A new question')
        assert supplied.llm.calls == []
        assert await memory._method.save() == before
        await submit_episode(memory, episode('second', task='Another task', end=True))
        assert len(supplied.llm.calls) == 2
        assert 'FAILED TRIAL' in supplied.llm.calls[0][-1]['content']
        assert 'Here are the trials:' in supplied.llm.calls[1][-1]['content']
        state = await memory._method.save()
        assert state['rules_with_count'] == [['Verify the observation before choosing the final answer.', 2]]
        async with memory.read_only():
            used = await memory.search('A new question')
        assert await memory._method.save() == state
        assert any(item['memory_type'] == 'insight' for item in used.items)
        assert all(row['source_item_indices'] for row in used.context)
        rule = next(item for item in used.items if item['memory_type'] == 'insight')
        assert set(rule['source_event_ids']) == {'failure-event', 'success-event', 'second-event'}


@pytest.mark.asyncio
async def test_observe_completion_dedup_and_snapshot_keep_pending_batch(tmp_path):
    item = episode('pending')
    snapshot = tmp_path/'memory.json'
    supplied = services()
    async with Memory(config(tmp_path/'one'), services=supplied) as memory:
        event = item.events[0]
        from dataclasses import replace
        await add_event(memory, replace(event, episode_id=item.episode_id))
        await submit_episode(memory, replace(item, events=None))
        await submit_episode(memory, replace(item, events=None))
        assert len((await memory._method.save())['successes'][item.task]) == 1
        await memory.save(snapshot)
        assert supplied.llm.calls == []  # save preserves pending work, never compiles.
    restored_services = services()
    async with Memory(config(tmp_path/'two'), services=restored_services) as restored:
        await restored.load(snapshot)
        await submit_episode(restored, episode('tail', task='Second task', end=True))
        assert len(restored_services.llm.calls) == 1
        before = await restored._method.save()
        async with restored.read_only():
            await restored.search('Different task')
        assert await restored._method.save() == before
        await restored.reset()
        assert not (await restored.search('Different task')).items


@pytest.mark.asyncio
async def test_missing_feedback_is_rejected_before_model_or_storage_mutation(tmp_path):
    from dataclasses import replace
    supplied = services()
    async with Memory(config(tmp_path), services=supplied) as memory:
        before = await memory._method.save()
        from mematlas.core.errors import CapabilityMismatchError
        with pytest.raises(CapabilityMismatchError, match='success'):
            await submit_episode(memory, replace(episode('unknown'), success=None))
        assert supplied.llm.calls == [] and supplied.embedder.calls == []
        assert await memory._method.save() == before


@pytest.mark.asyncio
async def test_l2_shortest_success_and_public_top_k(tmp_path):
    supplied = services()
    async with Memory(config(tmp_path, num_fewshots=2), services=supplied) as memory:
        await submit_episode(memory, episode('long', task='distant task', text='Thought 1: '+('long ' * 20)))
        await submit_episode(memory, episode('short', task='distant task', text='Thought 1: short'))
        await submit_episode(memory, episode('near', task='near task', end=True))
        used = await memory.search('new task', top_k=1)
        trials = [item for item in used.items if item['memory_type'] == 'successful_trial']
        assert len(trials) == 1 and trials[0]['metadata']['task'] == 'near task'
        used = await memory.search('new task', top_k=2)
        distant = next(item for item in used.items if item.get('metadata', {}).get('task') == 'distant task')
        assert 'Thought 1: short' in distant['content']
        assert distant['source_event_ids'] == ['short-event']
        same_task = await memory.search('near task')
        assert all(item.get('metadata', {}).get('task') != 'near task' for item in same_task.items)


def test_qa_adds_only_real_batch_boundary_without_changing_answers():
    from mematlas.methods.expel.qa import ExpeLQA
    from mematlas.core.evaluation import BenchmarkCase
    value = episode('one')
    case = BenchmarkCase(case_id='case', query=value.task, reference='private-test-answer')
    assert ExpeLQA.prepare_episode(case, value, last_for_scope=False) == value
    final = ExpeLQA.prepare_episode(case, value, last_for_scope=True)
    assert final.extensions['expel']['end_of_training'] is True
    assert final.success is True and final.task == value.task and final.events == value.events
    assert 'private-test-answer' not in json.dumps(final.to_dict())


def test_plain_locomo_without_real_training_feedback_is_not_silently_supported(tmp_path):
    from mematlas import ExperimentConfig, ExperimentRunner
    from mematlas.benchmarks.datasets import InMemoryBenchmark
    from mematlas.core.evaluation import BenchmarkCase
    supplied = services()
    supplied.objects['answer_llm'] = supplied.llm
    benchmark = InMemoryBenchmark('plain-history', [BenchmarkCase(case_id='one', query='Question?',
        history=[MemoryEvent('message', 'an unlabeled chat')], scope={'user_id': 'alice'})])
    experiment = ExperimentConfig(methods=(config(tmp_path),), benchmark=benchmark)
    report = ExperimentRunner().check(experiment, services=supplied)
    assert not report['ready'] and 'success' in json.dumps(report)
