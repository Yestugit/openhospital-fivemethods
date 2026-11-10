"""Real native HotpotQA small validation via registered ExpeLQA and ExperimentRunner."""
from __future__ import annotations
import argparse
import asyncio
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import types
from typing import Sequence

from mematlas import ExperimentConfig,ExperimentRunner
from mematlas.benchmarks.hotpotqa import HotpotQABenchmark
from mematlas.benchmarks.datasets import InMemoryBenchmark
from mematlas.core.files import atomic_write_json
from mematlas.core.plugin import MethodServices
from mematlas.core.models import MemoryEpisode
from mematlas.core.provider_config import resolve_providers
from mematlas.core.telemetry import UsageLedger
from mematlas.benchmarks.hotpotqa_env import WikipediaDocstore
from .example import environment_config
from .qa import ExpeLQA
from .hotpotqa_qa import execute_hotpotqa
from .hotpotqa_prompts import SYSTEM_REFLECTION_INSTRUCTION

SAMPLE_URL='https://raw.githubusercontent.com/LeapLabTHU/ExpeL/e41ec9a24823e7b560c561ab191441b56d9bcefc/data/hotpotqa/hotpot-qa-distractor-sample.joblib'
SAMPLE_SHA256='b3e49c0aea1f8151fe307b3af2e7d8fa13271e2401a4c74b1e053137c37b9790'


def convert_upstream_sample(source: Path,destination: Path) -> None:
    """Only load the exact trusted archive, then export all original fields as JSON.

    Pandas 1.x Int64Index moved in 2.x. A temporary compatibility alias is used
    only during conversion; runtime benchmark/QA never read pickle or need pandas.
    """
    if hashlib.sha256(source.read_bytes()).hexdigest()!=SAMPLE_SHA256:
        raise ValueError('Upstream sample SHA-256 mismatch; refusing to unpickle')
    import pandas as pd
    import joblib
    name='pandas.core.indexes.numeric';old=sys.modules.get(name)
    shim=types.ModuleType(name);shim.Int64Index=shim.UInt64Index=shim.Float64Index=pd.Index
    try:
        sys.modules[name]=shim
        data=joblib.load(source).to_dict('records')
    finally:
        if old is None:sys.modules.pop(name,None)
        else:sys.modules[name]=old
    rows=json.loads(json.dumps(data,default=lambda v:v.tolist(),ensure_ascii=False))
    atomic_write_json(destination,rows)
    atomic_write_json(destination.with_suffix('.source.json'),{'url':SAMPLE_URL,'sha256':SAMPLE_SHA256,
        'json_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'records':len(rows),
        'conversion':'all rows and columns retained; no answer-based selection'})


def download_sample(destination: Path) -> None:
    import requests
    response=requests.get(SAMPLE_URL,timeout=30,stream=True);response.raise_for_status()
    chunks=[];size=0
    try:
        for chunk in response.iter_content(65536):
            size+=len(chunk)
            if size>2_000_000:raise ValueError('Sample exceeds 2 MB download limit')
            chunks.append(chunk)
    finally:response.close()
    data=b''.join(chunks)
    if hashlib.sha256(data).hexdigest()!=SAMPLE_SHA256:raise ValueError('Downloaded sample SHA mismatch')
    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data)


async def run(args):
    from mematlas import MemAtlasConfig
    from mematlas.providers import OpenAICompatibleLLM
    from mematlas.providers.instrumentation import MeteredService
    config=MemAtlasConfig.load(args.config) if args.config else environment_config()
    if config.method.method_id!='expel' or config.method.params.get('benchmark')!='hotpotqa':
        raise ValueError('Native HotpotQA entry requires expel benchmark=hotpotqa')
    llm=resolve_providers(config.method.params,config.providers)['llm']
    if 'response_format' in llm.options:raise ValueError('Remove JSON response_format for native actions/rules')
    benchmark=HotpotQABenchmark(args.data,train_count=args.train_tasks,test_count=args.test_tasks,max_steps=args.max_steps)
    initial=ExperimentConfig(methods=(config,),benchmark=benchmark,answer_model=llm,
        protocol='experience_train_test',output_dir=str(args.output/'experiment'))
    ExpeLQA.validate_cases(initial,tuple(benchmark.cases('train')),tuple(benchmark.cases('test')))
    from mematlas import Memory
    report=Memory.check(config)
    if not report['ready']:raise ValueError(f'ExpeL not ready: {report}')
    args.output.mkdir(parents=True,exist_ok=False)
    config=replace(config,runtime=replace(config.runtime,artifact_dir=str(args.output/'memory')))
    atomic_write_json(args.output/'memory-config.json',config.to_dict())
    atomic_write_json(args.output/'benchmark.json',benchmark.describe())
    source=args.resume_from
    records=[]
    if source:
        previous=json.loads((source/'benchmark.json').read_text(encoding='utf-8'))
        if previous!=benchmark.describe():raise ValueError('Resume requires identical benchmark/data/budget/splits')
        previous_records=json.loads((source/'trajectories.json').read_text(encoding='utf-8'))
        records=[r for r in previous_records if r.get('episode') is not None]
        atomic_write_json(args.output/'prior-infrastructure-errors.json',[r for r in previous_records if r.get('episode') is None])
        atomic_write_json(args.output/'resume.json',{'source':str(source.resolve()),'policy':'public runner resume; reuse actual training episodes and cached test results'})
    def sink(record):
        records.append(record);atomic_write_json(args.output/'trajectories.json',records)
        success=record['episode']['success'] if record['episode'] else record['status']
        print(f"{record['split']} {record['case_id']}: success={success}, actions={len(record['turns'])}",flush=True)
    training_usage=UsageLedger()
    actor=OpenAICompatibleLLM(model=llm.model,base_url=llm.base_url,api_key_env=llm.api_key_env,
        default_options=llm.options,usage=training_usage)
    metered=MeteredService(actor,training_usage,'llm')
    supplied=MethodServices(objects={'answer_llm':metered,'hotpotqa_trace_sink':sink,
        'hotpotqa_docstore_factory':lambda:WikipediaDocstore(args.output/'wikipedia-snapshots')})
    qa=ExpeLQA(initial,supplied);training=[];reflection_records=[]
    try:
        if source:
            cases={case.case_id:case for case in benchmark.cases('train')}
            for row in json.loads((source/'training-episodes.json').read_text(encoding='utf-8')):
                original=row['case_id'].rsplit(':attempt-',1)[0]
                training.append(replace(cases[original],case_id=row['case_id'],episode=MemoryEpisode.from_dict(row['episode'])))
            atomic_write_json(args.output/'training-episodes.json',[dict(case_id=c.case_id,episode=c.episode.to_dict()) for c in training])
            for name in ('training-usage.json','reflections.json'):
                path=source/name
                if path.exists():atomic_write_json(args.output/('previous-'+name),json.loads(path.read_text(encoding='utf-8')))
            # Preserve successful response snapshots; missing requests remain genuinely online.
            import shutil
            shutil.copytree(source/'wikipedia-snapshots',args.output/'wikipedia-snapshots',dirs_exist_ok=True)
        for case in (() if source else benchmark.cases('train')):
            reflections=[]
            for attempt in range(args.train_attempts):
                episode=await execute_hotpotqa(qa,case,None,reflections=reflections)
                identity=case.case_id+f':attempt-{attempt+1}'
                training.append(replace(case,case_id=identity,episode=episode))
                atomic_write_json(args.output/'training-episodes.json',[dict(case_id=c.case_id,episode=c.episode.to_dict()) for c in training])
                if episode.success or attempt+1>=args.train_attempts:break
                # Reflect only on actual observed failure; reference answer is never provided.
                messages=[{'role':'system','content':SYSTEM_REFLECTION_INSTRUCTION},
                    {'role':'user','content':'Question: '+str(case.query)+'\n'+'\n'.join(e.content for e in episode.events)}]
                reflection=await metered.complete(messages,temperature=0,max_tokens=250)
                reflections.append(str(reflection))
                reflection_records.append(dict(case_id=case.case_id,messages=messages,output=str(reflection)))
                atomic_write_json(args.output/'reflections.json',reflection_records)
        atomic_write_json(args.output/'training-usage.json',{'summary':training_usage.summary(),'records':training_usage.to_dicts()})
        # Training execution happened externally; public runner consumes genuine labelled episodes.
        # Its registered ExpeLQA then performs test-time multi-step tool execution.
        supplied.objects['answer_llm']=actor
        actor.usage=None  # Evaluation uses the runner's active ledger, not the collection ledger.
        experiment=replace(initial,methods=(config,),train_benchmark=InMemoryBenchmark('hotpotqa_actual_training',training))
        report=ExperimentRunner().check(experiment,services=supplied)
        atomic_write_json(args.output/'preflight.json',report)
        if not report['ready']:raise ValueError(f'Native experiment preflight failed: {report}')
        result=(await ExperimentRunner().evaluate(experiment,services=supplied,
            resume_from=source/'experiment' if source else None))[0]
        atomic_write_json(args.output/'result.json',result.to_dict())
        summary={'benchmark':'hotpotqa','environment':'online Wikipedia Docstore',
            'train_task_count':args.train_tasks,'actual_train_episodes':len(training),
            'train_successes':sum(c.episode.success for c in training),
            'test_tasks':len(result.predictions),'metrics':dict(result.metrics),'failures':list(result.failures),
            'registered_qa':'mematlas.methods.expel.qa:ExpeLQA','paper_reproduction':False}
        atomic_write_json(args.output/'summary.json',summary)
        print(json.dumps(summary,ensure_ascii=False),flush=True)
        if result.failures:raise RuntimeError('Native experiment has execution failures; inspect preserved artifacts')
    finally:
        atomic_write_json(args.output/'training-usage.json',{'summary':training_usage.summary(),'records':training_usage.to_dicts()})
        if getattr(actor,'_client',None) is not None:await actor._client.close()


def main(argv: Sequence[str]|None=None):
    parser=argparse.ArgumentParser(description='ExpeL native HotpotQA: real Search/Lookup/Finish, actual training feedback, frozen evaluation')
    parser.add_argument('--data',type=Path,required=True,help='Plain HotpotQA JSON, or conversion output')
    parser.add_argument('--output',type=Path,help='New evidence directory, required unless converting only')
    parser.add_argument('--config',type=Path)
    parser.add_argument('--env-file',type=Path)
    parser.add_argument('--resume-from',type=Path,help='Reuse actual training and public checkpoint; an uncommitted test batch is rerun in full')
    parser.add_argument('--download-sample',action='store_true',help='Download exact pinned 745 KB archive and convert to --data')
    parser.add_argument('--convert-archive',type=Path,help='Convert exact pinned legacy archive to --data')
    parser.add_argument('--prepare-only',action='store_true',help='Prepare data without model calls')
    parser.add_argument('--train-tasks',type=int,default=2)
    parser.add_argument('--test-tasks',type=int,default=3)
    parser.add_argument('--train-attempts',type=int,choices=(1,2,3,4),default=2)
    parser.add_argument('--max-steps',type=int,default=7)
    args=parser.parse_args(argv)
    if args.output is None and not args.prepare_only:parser.error('--output is required')
    if args.env_file:
        from dotenv import load_dotenv
        if not args.env_file.is_file():parser.error('--env-file must exist')
        load_dotenv(args.env_file,override=True)
    if args.download_sample:
        if args.data.exists():parser.error('--data already exists; refuse to overwrite')
        archive=args.data.with_suffix('.joblib');download_sample(archive);convert_upstream_sample(archive,args.data)
    elif args.convert_archive:
        if args.data.exists():parser.error('--data already exists; refuse to overwrite')
        convert_upstream_sample(args.convert_archive,args.data)
    if not args.prepare_only:asyncio.run(run(args))


if __name__=='__main__':main()
