"""Audit native Mind2Web logs against pinned eval_sample and public Memory restore.

No model calls: reuse recorded raw outputs as the upstream actor's responses.
This checks protocol parity, not what a second independent model run would say.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
from copy import deepcopy
from dataclasses import replace
import json
import logging
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Sequence

from mematlas import MemAtlasConfig, Memory
from mematlas.benchmarks.mind2web import Mind2WebBenchmark
from mematlas.core.files import atomic_write_json
from mematlas.core.plugin import MethodServices
from mematlas.core.provenance import file_sha256, json_sha256
from mematlas.methods.awm.mind2web_qa import _token_counter
from mematlas.test.awm.mind2web_smoke import _env_helpers


class NoModel:
    async def complete(self, *args, **kwargs):
        raise AssertionError('Evidence audit must not call a model')


def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def upstream_functions():
    """Compile original functions without importing network-initializing modules."""
    root = Path(__file__).parent / 'reference/mind2web'
    namespace = {**vars(_env_helpers()), 'np': SimpleNamespace(sum=sum), 'Path': Path,
                 'json': json, 're': re, 'logger': logging.getLogger(__name__)}
    for path, names in ((root / 'utils/llm.py', {'extract_from_response'}),
                        (root / 'memory.py', {'eval_sample'})):
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


async def check(run_dir: Path, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    experiment = read(run_dir / 'experiment/experiment.json')
    data = Path(experiment['benchmark']['path'])
    benchmark = Mind2WebBenchmark(data)
    traces = read(run_dir / 'qa-traces.json')
    result = read(run_dir / 'result.json')
    assert not result['failures']
    state = read(run_dir / 'memory-readable.json')
    assert len(state['episodes']) == len(tuple(benchmark.cases('train')))
    history = state['induction_history']
    model_name = experiment['answer_model']['model']
    counter, _ = _token_counter(model_name)
    checks = []
    previous_workflow = ''
    training = list(benchmark.cases('train'))
    cases = [*training, *benchmark.cases('test')]
    namespace = upstream_functions()
    for index, case in enumerate(cases):
        trace = next(r for r in traces if r['case_id'] == case.case_id)
        generated = iter(r for r in trace['model_calls'] if r['status'] == 'predicted')
        oracle_messages = []
        def generate_response(*, messages, **kwargs):
            saved = next(generated)
            assert messages == saved['messages'], f'Upstream prompt differs for {case.case_id}'
            oracle_messages.append(deepcopy(messages))
            return saved['raw_output'], {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        concrete = benchmark.concrete_examples(case.case_id)
        exemplars = ([[{'role': 'user', 'content': previous_workflow}]] if previous_workflow else []) + concrete
        namespace.update(generate_response=generate_response,
                         get_exemplars=lambda args: exemplars,
                         num_tokens_from_messages=lambda messages, model: counter(messages),
                         MAX_TOKENS={model_name: benchmark.max_context_tokens})
        args = SimpleNamespace(model=model_name, top_k_elements=benchmark.top_k,
            previous_top_k_elements=benchmark.previous_top_k, log_dir=str(output / 'upstream'),
            benchmark='test_task', website=benchmark.rows[case.case_id]['website'],
            suffix='workflow', temperature=0.0)
        namespace['eval_sample'](index, args, benchmark.rows[case.case_id])
        oracle = read(output / 'upstream' / model_name / 'test_task' / args.website / 'workflow' / f'{index}.json')[-1]
        assert next(generated, None) is None
        for key in ('element_acc', 'action_f1', 'step_success'):
            assert oracle[key] == [step[key] for step in trace['step_scores']]
        checks.append({'case_id': case.case_id, 'identical_actor_messages': True,
                       'identical_native_step_scores': True, 'model_calls_replayed': len(oracle_messages),
                       'upstream_success': oracle['success'][0]})
        if index < len(training):
            episode = read(run_dir / 'training-episodes.json')[case.case_id]
            assert episode['extensions']['awm']['canonical'] is False
            # The serialized online source events must match the actual last
            # actor input and response, not teacher-forced canonical actions.
            observations = [e['content'] for e in episode['events'] if e['kind'] == 'observation']
            actions = [e['content'] for e in episode['events'] if e['kind'] == 'action']
            records = [r for r in trace['model_calls'] if r['status'] == 'predicted']
            assert observations == [r['messages'][-1]['content'] for r in records]
            assert actions == [r['raw_output'] for r in records]
            # This bounded validation induces after each train task. Empty
            # experiences would not produce an induction_history entry.
            if actions:
                previous_workflow = history[index]['filtered']
    for point in result['learning_curve']:
        assert point['frozen_verified'] is True
    snapshots = run_dir / 'experiment/runs/001-awm/snapshots'
    recovery = read(snapshots / f'learned_{len(training)}.json')
    snapshot_path = snapshots / recovery['memory_file']
    snapshot = read(snapshot_path)
    digest = snapshot.pop('payload_sha256')
    assert digest == json_sha256(snapshot)
    assert snapshot['state'] == state
    cfg = MemAtlasConfig.load(run_dir / 'memory-config.json')
    cfg = replace(cfg, runtime=replace(cfg.runtime, artifact_dir=str(output / 'restore-runtime')))
    supplied = MethodServices(llm=NoModel())
    async with Memory(cfg, services=supplied) as memory:
        await memory.load(snapshot_path)
        before_path = output / 'restored-before.json'
        await memory.save(before_path)
        before = read(before_path)['state']
        assert before == state
        async with memory.read_only():
            contribution = await memory.search('audit', scope=cases[0].scope,
                extensions={'awm': {'website': args.website}})
        await memory.save(output / 'restored-after.json')
        assert read(output / 'restored-after.json')['state'] == state
        assert contribution.items[0]['content'] == state['workflow_text'][args.website]
        atomic_write_json(output / 'retrieved-memory.json', contribution.to_dict())
        await memory.reset(scope=cases[0].scope)
        await memory.save(output / 'reset.json')
        assert not read(output / 'reset.json')['state']['workflow_text']
    report = {'model_calls': 0, 'run_dir': str(run_dir.resolve()), 'cases': checks,
              'training_uses_actual_actor_input_and_output': True,
              'public_restore_equal': True, 'frozen_retrieval_equal': True, 'reset_empty': True,
              'frozen_test_verified': True,
              'upstream_comparison': 'original eval_sample with recorded model responses and same selected exemplars; no fresh model inference',
              'result_sha256': file_sha256(run_dir / 'result.json')}
    atomic_write_json(output / 'evidence-check.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New audit directory')
    args = parser.parse_args(argv)
    asyncio.run(check(args.run, args.output))


if __name__ == '__main__':
    main()
