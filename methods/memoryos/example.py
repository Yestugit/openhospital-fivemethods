"""给已有异步 Agent 接入 MemoryOS。使用要求查看 help(mematlas.methods.memoryos)。"""
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from mematlas import MemAtlasConfig, Memory, MemoryEpisode, MemoryEvent, MemoryUnit
from . import MemoryOSConfig


async def run(
    tasks: Sequence[str],
    agent_call: Callable[[list[dict[str, str]]], Awaitable[str]],
    *,
    providers: Mapping[str, Any],
) -> list[str]:
    """providers 填记忆所用模型；agent_call 沿用用户已有的 Agent。"""
    config = MemAtlasConfig(method=MemoryOSConfig(), providers=providers,
                            scope={"user_id": "alice"})
    answers = []
    async with Memory(config) as memory:
        for task in tasks:
            events = [MemoryEvent("message", task, role="user")]
            used = await memory.search(task)
            # 使用方法整理的分层上下文，回传完整贡献以保留来源。
            text = "\n\n".join(row["content"] for row in used.context)
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
        # 写入时按容量和热度自动迁移、演化，无需额外调用。
    return answers
