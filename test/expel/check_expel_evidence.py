"""Verify actual demo artifacts offline; python -m mematlas.test.expel.check_expel_evidence --help."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Sequence

from mematlas.core.files import atomic_write_json
from mematlas.core.provenance import json_sha256
from mematlas.methods.expel.core import parse_rules, update_rules


def load(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def snapshot(path: Path):
    value = load(path)
    checksum = value.pop('payload_sha256')
    assert checksum == json_sha256(value), f'Invalid checkpoint checksum: {path.name}'
    return value['state']


def check(root: Path) -> dict:
    state = snapshot(root/'memory.json')
    assert state == snapshot(root/'after-frozen.json') == snapshot(root/'restored.json')
    assert set(state['seen_episodes']) == {'train-1','train-2','train-3'}
    trials = load(root/'agent-trials.json')
    for record in trials:
        episode = record['episode']
        feedback = json.loads(episode['feedback'])
        assert episode['success'] == (feedback['finished'] and episode['output']==feedback['expected'])
        assert not episode['applied_memory']  # Source train phase does not dynamically retrieve.
        assert record['model_calls'] and episode['events']
    assert sum(t['episode']['success'] for t in trials)==2
    assert snapshot(root/'after-learn-1.json')['rules_with_count']==[]
    assert snapshot(root/'after-learn-2.json')['rules_with_count']==[]
    assert [h['stage'] for h in state['critique_history']]==['compare','success']
    rules=[]
    all_sources={e['event_id'] for record in trials for e in record['episode']['events']}
    for history in state['critique_history']:
        assert [list(r) for r in rules]==history['before']
        operations=parse_rules(history['output'])
        assert [list(row) for row in operations]==history['operations']
        rules=update_rules(rules,operations,list_full=len(rules)>=25)
        assert [list(r) for r in rules]==history['after']
        assert set(history['source_event_ids']) <= all_sources
    assert [list(r) for r in rules]==state['rules_with_count']
    for ids in state['rule_sources'].values():
        assert ids and set(ids) <= all_sources
    used=load(root/'retrieval.json')
    assert used==load(root/'restored-retrieval.json')
    assert sum(i['memory_type']=='successful_trial' for i in used['items'])==2
    for item in used['items']:
        assert set(item['source_event_ids']) <= all_sources
    evaluation=load(root/'evaluation.json')
    assert evaluation['success'] == (evaluation['feedback']['finished'] and
        evaluation['output']==evaluation['feedback']['expected'])
    assert evaluation['applied_memory']==[used]
    cleared=snapshot(root/'after-reset.json')
    assert not any(cleared[k] for k in ('documents','successes','failures','rules_with_count','seen_episodes'))
    import faiss
    import numpy as np
    index_path=root/'expel.faiss'
    if not index_path.exists():
        index_path=next((root/'runtime').rglob('expel.faiss'))
    # Native FAISS file APIs on Windows cannot open the Chinese evidence directory.
    index=faiss.deserialize_index(np.frombuffer(index_path.read_bytes(),dtype='uint8'))
    assert index.ntotal==len(state['documents']) and index.d==1024
    readable=deepcopy(state)
    for document in readable['documents']:
        document['embedding_dimensions']=len(document.pop('embedding'))
    atomic_write_json(root/'memory-readable.json',readable)
    result=dict(checkpoint_checksums=True,training_labels_checked=True,critique_replayed=True,
        source_lineage_checked=True,frozen_state_equal=True,restored_state_and_retrieval_equal=True,
        test_not_learned=True,reset_empty=True,faiss_document_count=index.ntotal,embedding_dimensions=index.d,
        rules=state['rules_with_count'],official_benchmark_acceptance='not evaluated',human_review='pending')
    atomic_write_json(root/'evidence-check.json',result)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser=argparse.ArgumentParser(description='Offline verify ExpeL real-demo evidence and export readable memory')
    parser.add_argument('run_dir',type=Path)
    args=parser.parse_args(argv)
    print(json.dumps(check(args.run_dir),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
