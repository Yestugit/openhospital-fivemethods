"""给已有异步 Agent 接入 Mem0。配置与数据要求查看 help(mematlas.methods.mem0)。"""
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from mematlas import MemAtlasConfig, Memory, MemoryEpisode, MemoryEvent, MemoryUnit, RuntimeConfig
from . import Mem0Config


async def run(
    tasks: Sequence[str],
    agent_call: Callable[[list[dict[str, str]]], Awaitable[str]],
    *,
    providers: Mapping[str, Any],
    python_executable: str | None = None,
) -> list[str]:
    """providers 填记忆所用模型；agent_call 沿用用户已有的 Agent。"""
    # 独立环境安装 requirements-lock.txt；embedding provider 指定 dimensions。
    runtime = RuntimeConfig(executor="subprocess", python_executable=python_executable) if python_executable else RuntimeConfig()
    config = MemAtlasConfig(method=Mem0Config(top_k=3), providers=providers,
                            scope={"user_id": "alice"}, runtime=runtime)
    answers = []
    async with Memory(config) as memory:
        for task in tasks:
            events = [MemoryEvent("message", task, role="user")]
            used = await memory.search(task)
            # 将全部返回条目提供给 Agent，回传时保留完整贡献与来源。
            text = "\n".join(item["content"] for item in used.items)
            answer = await agent_call([
                {"role": "system", "content": f"相关记忆：\n{text}"},
                {"role": "user", "content": task},
            ])
            events.append(MemoryEvent("message", answer, role="assistant"))
            unit = MemoryUnit.from_episode(MemoryEpisode(
                task=task, output=answer, events=events, applied_memory=[used],
            ))
            receipt = await memory.add(unit)
            await memory.episode_end(receipt.episode_ref)
            answers.append(answer)
    return answers
