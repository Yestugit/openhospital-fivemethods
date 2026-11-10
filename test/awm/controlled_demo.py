"""Controlled service/Agent integration evidence only. Never invokes a benchmark or API."""
from __future__ import annotations
import argparse
import asyncio
from dataclasses import replace
import json
from pathlib import Path

from mematlas import MemAtlasConfig,Memory
from mematlas.core.files import atomic_write_json
from mematlas.methods.awm.example import ingest
from mematlas.test.awm.awm_fixtures import config,services,episode


async def check(output:Path):
    supplied=services()
    cfg=MemAtlasConfig.from_dict(config(output/'unused',profile='webarena'))
    episodes=[episode('failed',success=False),episode('one'),episode('two',template_id=2)]
    await ingest(cfg,episodes,output,services=supplied)
    atomic_write_json(output/'controlled-model-calls.json',supplied.llm.calls)
    state=json.loads((output/'after-3.json').read_text(encoding='utf-8'))['state']
    frozen=json.loads((output/'after-frozen.json').read_text(encoding='utf-8'))['state']
    assert state==frozen
    restored_cfg=replace(cfg,runtime=replace(cfg.runtime,artifact_dir=str(output/'restore-runtime')))
    async with Memory(restored_cfg,services=services()) as memory:
        await memory.load(output/'after-3.json')
        await memory.save(output/'restored.json')
        assert json.loads((output/'restored.json').read_text(encoding='utf-8'))['state']==state
        await memory.reset();await memory.save(output/'after-reset.json')
    atomic_write_json(output/'memory-readable.json',state)
    result={'kind':'controlled models and synthetic navigation traces; integration only',
        'benchmark_evaluation_executed':False,'real_model_calls':0,'quality_claim':False,
        'induction_calls':len(supplied.llm.calls),'stored_episodes':len(state['episodes']),
        'workflow_files':len(state['workflow_text']),'frozen_equal':True,'public_restore_equal':True,
        'human_review':'pending'}
    atomic_write_json(output/'checks.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description='AWM controlled integration checks only; no benchmark or model API')
    parser.add_argument('--output',type=Path,required=True)
    asyncio.run(check(parser.parse_args().output))


if __name__=='__main__':main()
