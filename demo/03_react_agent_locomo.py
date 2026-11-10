"""用自己的 ReAct + reflection Agent 测评 LoCoMo 前两题。

项目根目录安装：python -m pip install -e ".[openai,benchmarks]" faiss-cpu
设置 DASHSCOPE_API_KEY 后运行：python src/mematlas/demo/03_react_agent_locomo.py
数据：agent-memory-benchmarks/LoCoMo/data/locomo10.json。两题共享完整历史。
为快速演示 Agent 接入，短期容量设为 256，容纳本会话后直接检索答题。
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4

from mematlas import (ExperimentConfig, ExperimentRunner, MemAtlasConfig,
                      MemoryEpisode, MemoryEvent, RuntimeConfig)
from mematlas.core.contributions import contribution_text
from mematlas.core.models import content_text
from mematlas.providers import OpenAICompatibleLLM


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "agent-memory-benchmarks/LoCoMo/data/locomo10.json"
os.environ.setdefault("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL = "qwen-plus"


class ReActAgent:
    """把现有 Agent 整理成 async agent(case, memory) -> MemoryEpisode。"""

    def __init__(self):
        self.llm = OpenAICompatibleLLM(MODEL, base_url=None, api_key_env="DASHSCOPE_API_KEY")
        self.steps = {}

    async def __call__(self, case, memory):
        task, notes, used = content_text(case.query), [], []
        events = [MemoryEvent("message", task, role="user")]
        for step in range(2):
            evidence = "\n\n".join(notes)
            choices = ('Choose retrieve_memory.' if step == 0 else
                       'Choose retrieve_memory if evidence is insufficient; otherwise choose finish.')
            decision = await self.llm.complete([
                {"role": "system", "content": "You are a ReAct agent. Select the next action."},
                {"role": "user", "content": f"Question: {task}\nEvidence: {evidence}\n{choices}\n"
                 'Return JSON: {"action":"retrieve_memory","query":"search query"} '
                 'or {"action":"finish"}.'},
            ], temperature=0, response_format={"type": "json_object"})
            action = json.loads(decision)
            events.append(MemoryEvent("message", decision, role="assistant"))
            if step and action.get("action") == "finish":
                break
            if action.get("action") != "retrieve_memory":
                raise ValueError(f"Unsupported Agent action: {action}")
            query = str(action.get("query") or task)
            # 工具由 Memory 提供；完整贡献随经历返回，保留实际使用来源。
            contribution = await memory.search(query, scope=case.scope)
            used.append(contribution)
            observation = contribution_text(contribution) or "No related memory found."
            events.append(MemoryEvent("message", f"Tool observation: {observation}", role="user"))
            reflection = await self.llm.complete([
                {"role": "system", "content": "Reflect briefly on this retrieval step using only its evidence."},
                {"role": "user", "content": f"Question: {task}\nSearch: {query}\nObservation: {observation}\n"
                 "State the supported facts and any missing evidence in at most two sentences."},
            ], temperature=0)
            events.append(MemoryEvent("message", f"Reflection: {reflection}", role="assistant"))
            notes.append(f"Observation: {observation}\nReflection: {reflection}")
        evidence = "\n\n".join(notes)
        answer = await self.llm.complete([
            {"role": "system", "content": "Answer using only retrieved evidence. Give only the requested answer. "
             'If unsupported, reply exactly "No information available."'},
            {"role": "user", "content": f"Question: {task}\nEvidence and reflections:\n{evidence}"},
        ], temperature=0)
        events.append(MemoryEvent("message", answer, role="assistant"))
        self.steps[case.case_id] = len(used)
        # 测试答案仅用于评分；Runner 负责导入历史，这里不提交学习。
        return MemoryEpisode(task=task, scope=case.scope, output=answer,
                             events=events, applied_memory=used)


async def main():
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise ValueError("请先设置 DASHSCOPE_API_KEY")
    if not DATA.is_file():
        raise FileNotFoundError(f"请准备 LoCoMo 原始数据：{DATA}")
    run_dir = ROOT / ".mematlas/demo-runs" / f"03-react-locomo-{uuid4().hex[:8]}"
    llm = {"model": MODEL, "base_url": None, "api_key_env": "DASHSCOPE_API_KEY"}
    memory = MemAtlasConfig(
        method={"method_id": "memoryos", "params": {"short_term_capacity": 256}},
        providers={"llm": llm, "embedding": {"model": "text-embedding-v4", "base_url": None,
                  "dimensions": 1024, "api_key_env": "DASHSCOPE_API_KEY"}},
        runtime=RuntimeConfig(artifact_dir=str(run_dir / "memory")),
    )
    experiment = ExperimentConfig(
        methods=(memory,), benchmark={"kind": "locomo", "path": str(DATA)},
        test_limit=2, output_dir=str(run_dir / "evaluation"),
    )
    agent = ReActAgent()
    try:
        result = (await ExperimentRunner().evaluate(experiment, agent=agent))[0]
        for prediction in result.predictions:
            print(f"{prediction.case_id}：{prediction.output}")
            print("反思步数：", agent.steps.get(prediction.case_id, 0))
        print("指标：", dict(result.metrics))
        print("失败数：", len(result.failures))
        print("结果目录：", result.run_metadata.get("output_dir", experiment.output_dir))
        if result.failures or len(result.predictions) != 2:
            raise RuntimeError(f"两题测评未全部完成：{result.failures}")
    finally:
        await agent.llm._client.close()


if __name__ == "__main__":
    asyncio.run(main())
