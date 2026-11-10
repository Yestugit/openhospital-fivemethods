"""终端连续对话：ReAct + 每轮 reflection 接入 MemoryOS。

项目根目录安装：python -m pip install -e ".[openai]" faiss-cpu
设置 DASHSCOPE_API_KEY 后运行：python src/mematlas/demo/01_react_agent_memory.py
输入 exit 结束；每次启动创建独立记忆，启动时打印存储目录。
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4

from mematlas import MemAtlasConfig, Memory, MemoryEpisode, MemoryEvent, RuntimeConfig, MemoryUnit
from mematlas.core.models import content_text
from mematlas.methods.memoryos import MemoryOSConfig
from mematlas.providers import OpenAICompatibleLLM


ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL = os.getenv("QWEN_MODEL", "qwen-plus")


def providers():
    return {
        "llm": {"model": MODEL, "base_url": None, "api_key_env": "DASHSCOPE_API_KEY"},
        "embedding": {"model": "text-embedding-v4", "base_url": None,
                      "dimensions": 1024, "api_key_env": "DASHSCOPE_API_KEY"},
    }


def memory_text(contribution):
    rows = contribution.context or contribution.items
    return "\n".join(content_text(row["content"]) for row in rows)


def parse_action(text):
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return {}


class ReActAgent:
    """One or two ReAct retrieval steps, each followed by a reflection."""

    def __init__(self):
        self.llm = OpenAICompatibleLLM(MODEL, api_key_env="DASHSCOPE_API_KEY")
        self.config = {"model": MODEL, "max_react_steps": 2, "reflection": "after_each_step"}
        self.trace = []

    async def answer(self, task, memory, *, scope=None):
        self.trace = []
        notes = []
        contributions = []
        for step in range(2):
            instruction = (
                'Return JSON only: {"action":"retrieve_memory","query":"..."}. '
                "Choose a short memory search query for the task."
                if step == 0 else
                'Return JSON only: {"action":"retrieve_memory","query":"..."} or {"action":"finish"}. '
                "Use another search only if the current evidence is insufficient."
            )
            previous = "\n".join(notes)
            decision = await self.llm.complete([
                {"role": "system", "content": "You are a concise ReAct agent."},
                {"role": "user", "content": f"Task: {task}\nPrevious observations:\n{previous}\n{instruction}"},
            ], temperature=0)
            action = parse_action(decision)
            if step and action.get("action") != "retrieve_memory":
                break
            query = str(action.get("query") or task)
            contribution = await memory.search(query, scope=scope)
            contributions.append(contribution)
            observation = memory_text(contribution) or "No related memory found."
            reflection = await self.llm.complete([
                {"role": "system", "content": "Reflect on the last ReAct step in one sentence."},
                {"role": "user", "content": f"Task: {task}\nAction: retrieve_memory({query})\n"
                 f"Observation: {observation}\nWhat did this step establish, and is more retrieval needed?"},
            ], temperature=0)
            notes.append(f"Observation: {observation}\nReflection: {reflection}")
            self.trace.append({"action": query, "observation": observation, "reflection": reflection})

        steps = "\n".join(notes)
        answer = await self.llm.complete([
            {"role": "system", "content": "You are a helpful conversational assistant. Respond to the current "
             "message using relevant retrieved memory. Acknowledge new information; do not invent past facts. "
             "If the user states a plan or fact without a question, simply acknowledge it. "
             "Reflections are auxiliary and can be mistaken. Be concise and use the user's language."},
            {"role": "system", "content": f"Retrieved memory and reflections:\n{steps}"},
            {"role": "user", "content": task},
        ], temperature=0)
        return answer, contributions

    async def close(self):
        await self.llm._client.close()


async def main():
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise RuntimeError("请先设置环境变量 DASHSCOPE_API_KEY")
    run_dir = ROOT / ".mematlas" / "demo-runs" / f"01-agent-{uuid4().hex[:8]}"
    config = MemAtlasConfig(
        method=MemoryOSConfig(),
        providers=providers(),
        scope={"user_id": "demo-user", "agent_id": "react-agent"},
        runtime=RuntimeConfig(artifact_dir=str(run_dir)),
    )
    agent = ReActAgent()
    try:
        async with Memory(config) as memory:
            storage_dir = memory.services.artifact_dir / "methods" / "memoryos"
            print("开始对话，输入 exit 结束。")
            print("记忆目录：", storage_dir)
            print("运行记录：", memory.services.artifact_dir)
            while True:
                try:
                    question = input("\n你：").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if question.lower() == "exit":
                    break
                if not question:
                    continue
                user_event = MemoryEvent("message", question, role="user")
                answer, used = await agent.answer(question, memory, scope=config.scope)
                events = [user_event, MemoryEvent("message", answer, role="assistant")]
                unit = MemoryUnit.from_episode(MemoryEpisode(task=question, output=answer, events=events,
                                                   applied_memory=used))
                receipt = await memory.add(unit)
                await memory.episode_end(receipt.episode_ref)
                print("Agent：", answer)
                for index, step in enumerate(agent.trace, 1):
                    print(f"第 {index} 轮反思：", step["reflection"])
            print("\n对话结束，记忆保存在：", storage_dir)
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
