"""Run a ranked Mind2Web manifest through the registered AWMQA and ExperimentRunner.

Example: python -m mematlas.methods.awm.mind2web_example --data manifest.json
         --env-file .env.local --output new-run
The output keeps native scores, complete episodes, model prompts/responses,
workflow state, recovery checkpoints and runtime operation journals.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Sequence

from mematlas import ExperimentConfig, ExperimentRunner, MemAtlasConfig
from mematlas.benchmarks.mind2web import Mind2WebBenchmark
from mematlas.core.files import atomic_write_json
from mematlas.core.plugin import MethodServices
from mematlas.core.provenance import redact_sensitive
from mematlas.core.provider_config import resolve_providers


def load_env(path: Path) -> None:
    """Read local provider settings without printing or persisting credentials."""
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        key, separator, value = line.partition('=')
        if separator and key.strip() and not key.lstrip().startswith('#'):
            os.environ.setdefault(key.strip(), value.strip().strip('\"\''))


async def run(args: argparse.Namespace):
    if args.env_file:
        load_env(args.env_file)
    benchmark = Mind2WebBenchmark(args.data)
    expected_mode = 'offline' if benchmark.training_mode == 'canonical' else 'online'
    if args.config:
        memory = MemAtlasConfig.load(args.config)
        if memory.method.method_id != 'awm' or memory.method.params['profile'] != 'mind2web':
            raise ValueError('Mind2Web example requires AWM profile=mind2web')
        if memory.method.params.get('induction_mode', 'online') != expected_mode:
            raise ValueError(f'Manifest training_mode={benchmark.training_mode} needs induction_mode={expected_mode}')
        provider = resolve_providers(memory.method.params, memory.providers)['llm']
    else:
        provider = {'model': os.getenv('AMEM_LLM_MODEL', 'qwen-flash'),
                    'base_url': os.getenv('AMEM_BASE_URL'), 'api_key_env': 'AMEM_API_KEY',
                    'options': {'extra_body': {'enable_thinking': False}}}
        memory = MemAtlasConfig.from_dict({
            'method': {'method_id': 'awm', 'params': {'profile': 'mind2web', 'induction_mode': expected_mode}},
            'providers': {'llm': provider}})
    output = args.output.resolve()
    memory = replace(memory, runtime=replace(memory.runtime, artifact_dir=str(output / 'runtime')))
    config = ExperimentConfig(methods=(memory,), benchmark={'kind': 'mind2web', 'path': str(args.data.resolve())},
        protocol='experience_train_test', answer_model=provider, output_dir=str(output / 'experiment'), seed=benchmark.seed)
    traces = []
    def sink(record):
        traces.append(record)
        atomic_write_json(output / 'qa-traces.json', traces)
        steps = record['step_scores']
        print(f"{record['case_id']}: steps={len(steps)}, "
              f"correct={sum(s['step_success'] for s in steps):g}, "
              f"candidate_misses={sum(s['reason'] == 'candidate_rank' for s in steps)}", flush=True)
    services = MethodServices(objects={'awm_trace_sink': sink})
    runner = ExperimentRunner()
    check = runner.check(config, services=services)
    if not check['ready']:
        raise ValueError('Mind2Web experiment not ready: ' + json.dumps(check, ensure_ascii=False))
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output / 'memory-config.json', redact_sensitive(memory.to_dict()))
    atomic_write_json(output / 'benchmark.json', benchmark.describe())
    result = (await runner.evaluate(config, services=services))[0]
    atomic_write_json(output / 'result.json', result.to_dict())
    # The runner's public recovery artifacts contain the exact frozen-test
    # starting state. Copy readable state; keep the complete original snapshot.
    snapshots = output / 'experiment/runs/001-awm/snapshots'
    trained = snapshots / f"learned_{len(tuple(benchmark.cases('train')))}.json"
    if trained.exists():
        recovery = json.loads(trained.read_text(encoding='utf-8'))
        snapshot = json.loads((snapshots / recovery['memory_file']).read_text(encoding='utf-8'))
        atomic_write_json(output / 'memory-readable.json', snapshot['state'])
        atomic_write_json(output / 'training-episodes.json', recovery['integration'].get('training_episodes', {}))
    summary = {'benchmark': benchmark.describe(), 'metrics': dict(result.metrics),
               'failures': list(result.failures), 'training_tasks': len(tuple(benchmark.cases('train'))),
               'test_tasks': len(tuple(benchmark.cases('test'))),
               'test_predictions': len(result.predictions),
               'registered_qa': 'mematlas.methods.awm.qa:AWMQA', 'live_browser': False}
    atomic_write_json(output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if result.failures:
        raise RuntimeError('Mind2Web execution failed; inspect result.json and qa-traces.json')
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--data', type=Path, required=True, help='Disjoint ranked train/test manifest JSON')
    result.add_argument('--output', type=Path, required=True, help='New directory for logs, results and memory')
    result.add_argument('--config', type=Path, help='Optional AWM Memory config; default uses AMEM_* settings')
    result.add_argument('--env-file', type=Path, help='Optional local provider environment file')
    return result


def main(argv: Sequence[str] | None = None) -> None:
    asyncio.run(run(parser().parse_args(argv)))


if __name__ == '__main__':
    main()
