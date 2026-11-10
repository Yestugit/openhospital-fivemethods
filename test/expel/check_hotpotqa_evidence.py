"""Offline audit of native HotpotQA artifacts, including public no-model load/save."""
from __future__ import annotations
import argparse
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from dataclasses import replace

from mematlas import MemAtlasConfig,Memory
from mematlas.core.files import atomic_write_json
from mematlas.core.provenance import json_sha256
from mematlas.core.plugin import MethodServices
from mematlas.benchmarks.hotpotqa import answer_scores
from mematlas.methods.expel.core import parse_rules,update_rules


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def checked_snapshot(path):
    value=read(path);digest=value.pop('payload_sha256')
    assert json_sha256(value)==digest,f'Invalid checksum: {path}'
    return value


class NoModel:
    async def complete(self,*args,**kwargs):raise AssertionError('Offline audit cannot call models')
    async def embed(self,*args,**kwargs):raise AssertionError('Offline audit cannot call models')


async def restore_check(root,snapshot):
    config=MemAtlasConfig.from_dict(read(root/'memory-config.json'))
    with tempfile.TemporaryDirectory(prefix='expel-hotpotqa-audit-') as directory:
        temporary=Path(directory)
        config=replace(config,runtime=replace(config.runtime,artifact_dir=str(temporary/'memory')))
        async with Memory(config,services=MethodServices(llm=NoModel(),embedder=NoModel())) as memory:
            await memory.load(snapshot)
            await memory.save(temporary/'restored.json')
        before,after=checked_snapshot(snapshot),checked_snapshot(temporary/'restored.json')
        assert before['state']==after['state'] and before['history']==after['history']


def check(root):
    result=read(root/'result.json');assert not result['failures']
    curve=result['learning_curve'];assert curve and curve[-1]['frozen_verified'] is True
    records=read(root/'trajectories.json');assert all(r.get('episode') is not None for r in records)
    train=[r for r in records if r['split']=='train']
    # A resumed uncommitted test batch may repeat completed calls. Select exactly
    # the episode committed by Prediction metadata; retain superseded raw traces.
    committed_ids={p['metadata']['episode_id'] for p in result['predictions']}
    tests=[r for r in records if r['split']=='test' and r['episode']['episode_id'] in committed_ids]
    assert len(tests)==3 and len({r['case_id'] for r in train})==2
    assert not {r['episode']['task'] for r in train}&{r['episode']['task'] for r in tests}
    train_ids={r['episode']['episode_id'] for r in train}
    sources={e['event_id'] for r in train for e in r['episode']['events']}
    for record in [*train,*tests]:
        episode=record['episode'];assert episode['events'] and record['turns']
        finish=next((t['action'][7:-1] for t in record['turns'] if t['action'].startswith('Finish[')),None)
        assert episode['output']==(finish or '')
        assert len(episode['events'])==2*len(record['turns'])
        for index,turn in enumerate(record['turns']):
            assert episode['events'][index*2+1]['content']==turn['observation']
            for query in record['wikipedia_requests']:
                cached=root/'wikipedia-snapshots'/f'{query["snapshot"]}.json'
                payload=read(cached)
                assert payload['url']==query['url']
                assert hashlib.sha256(json.dumps(query['query'],sort_keys=True).encode()).hexdigest()==query['snapshot']
        if record['split']=='train':assert not episode['applied_memory']
        else:
            assert episode['applied_memory']
            for contribution in episode['applied_memory']:
                for item in contribution['items']:assert set(item['source_event_ids'])<=sources
    prediction_by_id={p['case_id']:p for p in result['predictions']}
    for record in tests:
        prediction=prediction_by_id[record['case_id']]
        assert bool(answer_scores(prediction['output'],prediction['reference'])[0])==record['episode']['success']
        assert prediction['output']==record['episode']['output']
    snapshots=root/'experiment/runs/001-expel/snapshots'
    final=checked_snapshot(snapshots/f'learned_{len(train)}.json')
    memory=snapshots/final['memory_file']
    assert hashlib.sha256(memory.read_bytes()).hexdigest()==final['memory_sha256']
    state=checked_snapshot(memory)['state'];assert set(state['seen_episodes'])==train_ids
    saved_trials=[t for group in ('successes','failures') for trials in state[group].values() for t in trials]
    assert {t['id'] for t in saved_trials}==train_ids
    actual_by_id={r['episode']['episode_id']:r['episode'] for r in train}
    for trial in saved_trials:
        raw=actual_by_id[trial['id']]
        assert raw['success']==trial['success']
        assert trial['content']=='\n'.join(e['content'] for e in raw['events'])
    rules=[]
    for history in state['critique_history']:
        assert history['before']==[list(row) for row in rules]
        rules=update_rules(rules,parse_rules(history['output']),list_full=len(rules)>=25)
        assert history['after']==[list(row) for row in rules]
    assert state['rules_with_count']==[list(row) for row in rules]
    asyncio.run(restore_check(root,memory))
    readable=deepcopy(state)
    for doc in readable['documents']:doc['embedding_dimensions']=len(doc.pop('embedding'))
    atomic_write_json(root/'memory-readable.json',readable)
    report={'source_order_split':read(root/'benchmark.json')['split_policy'],'actual_train_episodes':len(train),
        'train_successes':sum(r['episode']['success'] for r in train),'test_successes':sum(r['episode']['success'] for r in tests),
        'test_count':len(tests),'superseded_test_attempts':sum(r['split']=='test' for r in records)-len(tests),
        'rule_count':len(rules),'vector_documents':len(state['documents']),
        'frozen_verified':True,'public_restore_equal':True,'restore_model_calls':0,
        'test_not_learned':True,'rule_operations_replayed':True,'snapshot_checksums_verified':True,
        'wikipedia_response_snapshots_verified':True,'metrics':result['metrics'],'human_review':'pending'}
    atomic_write_json(root/'evidence-check.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description='Offline verify ExpeL native HotpotQA actual episodes, Wikipedia snapshots and frozen state')
    parser.add_argument('run_dir',type=Path)
    print(json.dumps(check(parser.parse_args().run_dir),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
