"""ExpeL 真实模型接入示例；配置与数据要求查看 help(mematlas.methods.expel)。

使用本地虚构项目记录环境提供工具观察与成败反馈，演示学习、冻结检索和保存恢复。
参数查看 python -m mematlas.methods.expel.example --help。
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from pathlib import Path
import json
import os
import re
from typing import Sequence

from mematlas import MemAtlasConfig, Memory, MemoryEpisode, MemoryEvent, MemoryUnit
from mematlas.core.contributions import contribution_text
from mematlas.core.files import atomic_write_json
from mematlas.core.provider_config import resolve_providers
from mematlas.core.telemetry import UsageLedger
from .config import ExpeLConfig


def environment_config() -> MemAtlasConfig:
    """从 EXPEL_* 环境变量构建公共配置；缺省时沿用 AMEM_* 或示例默认值。"""
    def setting(name: str, default: str = '') -> str:
        return os.getenv('EXPEL_'+name, '').strip() or os.getenv('AMEM_'+name, '').strip() or default
    key_name = next((key for key in ('EXPEL_API_KEY','AMEM_API_KEY','DASHSCOPE_API_KEY') if os.getenv(key)), 'DASHSCOPE_API_KEY')
    embedding_key = next((key for key in ('EXPEL_EMBEDDING_API_KEY','AMEM_EMBEDDING_API_KEY') if os.getenv(key)), key_name)
    base = setting('BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
    return MemAtlasConfig(method=ExpeLConfig(benchmark='hotpotqa'), scope={'user_id':'expel-demo'}, providers={
        'llm': {'model':setting('LLM_MODEL','qwen-flash'), 'base_url':base, 'api_key_env':key_name,
                'options':{'extra_body':{'enable_thinking':False}, 'max_tokens':1800}},
        'embedding': {'model':setting('EMBEDDING_MODEL','text-embedding-v4'),
            'base_url':setting('EMBEDDING_BASE_URL',base), 'api_key_env':embedding_key,
            'dimensions':int(setting('EMBEDDING_DIMS','1024'))}})


async def execute_agent(task: str, key: str, records: dict[str,str], agent, *, budget: int,
                        memory_text: str = '') -> dict:
    """让 Agent 在本地记录环境中执行 Lookup/Finish，返回轨迹与环境评价。

    task 为任务文本，key 指定目标记录，records 提供环境数据；budget 限制动作次数。
    memory_text 为实际提供给 Agent 的记忆文本；成败由完成状态与记录值精确匹配决定。
    """
    messages = [dict(role='system', content=(
        'Use the authoritative synthetic project registry (local Docstore API) to answer the task. '
        'Project identifiers are fictional, not astronomical objects. Never use prior world knowledge. '
        'Never guess. Output a short Thought line and '
        'exactly one Action line per turn: Action N: Lookup[key] or Action N: Finish[answer]. '
        'Lookup returns the verified record. Finish terminates. The answer inside Finish must be '
        'the exact record value only, without a sentence or punctuation. Always include the Action N: '
        'prefix, including Finish actions. Do not generate Observation lines.\n'
        'Available key: '+key+'\nExperience references:\n'+memory_text)), dict(role='user',content=task)]
    events, calls, output, finished = [], [], '', False
    for step in range(1, budget+1):
        response = await agent.complete(messages, temperature=0, max_tokens=350)
        calls.append(dict(messages=[dict(m) for m in messages], response=response))
        matches = re.findall(r'\b(Lookup|Finish)\[([^\]\n]*)\]', response)
        # Like upstream LLM_PARSER, a single bare action gets its native Action prefix.
        # Preserve the raw response in calls and never change the answer or tool argument.
        formatted = response
        if len(matches) == 1 and not re.search(r'(?im)^Action\s*(?:\d+)?\s*:',response):
            formatted = f'Action {step}: {matches[0][0]}[{matches[0][1]}]'
        events.append(MemoryEvent('action', formatted, role='assistant'))
        messages.append(dict(role='assistant',content=response))
        if len(matches) != 1:
            observation = 'Invalid action: emit exactly one Lookup[key] or Finish[answer].'
        elif matches[0][0] == 'Finish':
            output, finished = matches[0][1].strip(), True
            break
        else:
            observation = records.get(matches[0][1].strip(), 'Record not found.')
        observed = f'Observation {step}: {observation}'
        events.append(MemoryEvent('tool_result', observed, role='tool'))
        messages.append(dict(role='user',content=observed))
    success = finished and output == records[key]
    return dict(task=task, events=events, output=output, success=success, calls=calls,
                feedback={'finished':finished,'budget':budget,'expected':records[key],
                          'reason':'exact environment match' if success else 'unfinished or incorrect answer'})


async def run(config: MemAtlasConfig, output: Path) -> dict:
    """执行三次模型任务，学习后冻结检索，再保存、恢复并清理恢复副本。

    config 提供 ExpeL 方法与模型服务，output 保存调用轨迹、经历、快照和统计。
    第一份经历将动作预算限制为一次，用于展示真实执行下的预算不足反馈。
    训练阶段不注入动态学习的规则和示例；任务数据来自本地演示环境。
    """
    if config.method.method_id != 'expel':
        raise ValueError('This example requires expel')
    config = replace(config, runtime=replace(config.runtime, artifact_dir=output/'runtime'))
    report = Memory.check(config)
    if not report['ready']:
        raise RuntimeError(f'ExpeL configuration not ready: {report}')
    llm = resolve_providers(config.method.params, config.providers)['llm']
    if 'response_format' in llm.options:
        raise ValueError('ExpeL emits plain rule operations; remove response_format from providers.llm.options')
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output/'config.json', config.to_dict())
    from mematlas.providers import OpenAICompatibleLLM
    agent_usage = UsageLedger()
    agent = OpenAICompatibleLLM(model=llm.model,base_url=llm.base_url,api_key_env=llm.api_key_env,
                               default_options=llm.options,usage=agent_usage)
    records = {'Orion location':'Kyoto','Vega location':'Lima','Sirius location':'Oslo'}
    task = 'What is the registered location of project Orion?'
    trials = [(task,'Orion location',1),(task,'Orion location',3),
              ('What is the registered location of project Vega?','Vega location',3)]
    trial_log = []
    try:
        async with Memory(config) as memory:
            for index, (question,key,budget) in enumerate(trials,1):
                value = await execute_agent(question,key,records,agent,budget=budget)
                identity = f'train-{index}'
                events = [replace(event, episode_id=identity) for event in value['events']]
                episode = MemoryEpisode(episode_id=identity,task=question,events=events,
                    output=value['output'],success=value['success'],
                    feedback=json.dumps(value['feedback'],ensure_ascii=False),applied_memory=[],
                    extensions={'expel':{'end_of_training':index==len(trials)}})
                await memory.save(output/f'before-learn-{index}.json')
                unit = MemoryUnit.from_episode(episode)
                receipt = await memory.add(unit)
                await memory.episode_end(receipt.episode_ref)
                await memory.save(output/f'after-learn-{index}.json')
                trial_log.append(dict(episode=episode.to_dict(),model_calls=value['calls'],
                    intervention='one-action budget' if index==1 else None))
                atomic_write_json(output/'agent-trials.json',trial_log)
                print(f'Train {index}: success={value["success"]}, budget={budget}',flush=True)
            checkpoint = output/'memory.json'
            await memory.save(checkpoint)
            async with memory.read_only():
                used = await memory.search('What is the registered location of project Sirius?',top_k=2)
                evaluated = await execute_agent('What is the registered location of project Sirius?',
                    'Sirius location',records,agent,budget=3,memory_text=contribution_text(used))
            await memory.save(output/'after-frozen.json')
            before = json.loads(checkpoint.read_text(encoding='utf-8'))['state']
            after = json.loads((output/'after-frozen.json').read_text(encoding='utf-8'))['state']
            if before != after:
                raise AssertionError('Frozen retrieval changed memory state')
            atomic_write_json(output/'retrieval.json',used.to_dict())
            atomic_write_json(output/'evaluation.json',dict(success=evaluated['success'],output=evaluated['output'],
                feedback=evaluated['feedback'],model_calls=evaluated['calls'],
                events=[event.to_dict() for event in evaluated['events']],applied_memory=[used.to_dict()]))
            atomic_write_json(output/'stats.json',memory.stats())
        restored_config = replace(config,runtime=replace(config.runtime,artifact_dir=output/'restored-runtime'))
        async with Memory(restored_config) as restored:
            await restored.load(checkpoint)
            await restored.save(output/'restored.json')
            restored_state = json.loads((output/'restored.json').read_text(encoding='utf-8'))['state']
            if restored_state != before:
                raise AssertionError('Public save/load changed state')
            async with restored.read_only():
                restored_used = await restored.search('What is the registered location of project Sirius?',top_k=2)
            atomic_write_json(output/'restored-retrieval.json',restored_used.to_dict())
            await restored.reset()
            await restored.save(output/'after-reset.json')
            if (await restored.search('new task')).items:
                raise AssertionError('Reset left memories')
        summary = dict(kind='real Qwen Agent / synthetic executable Docstore environment',
            quality_claim=False,train_successes=sum(t['episode']['success'] for t in trial_log),
            train_failures=sum(not t['episode']['success'] for t in trial_log),
            rule_count=len(before['rules_with_count']),critique_calls=len(before['critique_history']),
            document_count=len(before['documents']),frozen_unchanged=True,restore_equal=True,reset_empty=True,
            retrieval_rules=sum(i['memory_type']=='insight' for i in used.items),
            retrieval_examples=sum(i['memory_type']=='successful_trial' for i in used.items),
            evaluation_success=evaluated['success'])
        atomic_write_json(output/'summary.json',summary)
        atomic_write_json(output/'agent-usage.json',dict(summary=agent_usage.summary(),records=agent_usage.to_dicts()))
        return summary
    finally:
        if getattr(agent,'_client',None) is not None:
            await agent._client.close()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='ExpeL real-service Agent demo; three training trials and one frozen task')
    parser.add_argument('--config',type=Path,help='Public MemAtlas config; default EXPEL_*/AMEM_* environment')
    parser.add_argument('--env-file',type=Path,help='Local dotenv file, loaded only into this process')
    parser.add_argument('--output',type=Path,required=True,help='New evidence directory, never overwrite')
    args = parser.parse_args(argv)
    if args.env_file:
        if not args.env_file.is_file():
            parser.error('--env-file must exist')
        from dotenv import load_dotenv
        load_dotenv(args.env_file,override=True)
    cfg = MemAtlasConfig.load(args.config) if args.config else environment_config()
    print(asyncio.run(run(cfg,args.output)))


if __name__ == '__main__':
    main()
