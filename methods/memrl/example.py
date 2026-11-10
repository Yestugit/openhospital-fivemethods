"""给已有异步 Agent 接入 MemRL。使用要求查看 help(mematlas.methods.memrl)。

专有参数可省略，需要时直接填写字典或使用类型辅助：
    import mematlas.methods.memrl as memrl
    await memory.search(task, extensions={"memrl": memrl.SearchOptions(task_type="os")})
    MemoryEpisode(..., extensions={"memrl": {"task_id": 123}})
下方示例提供任务编号，并在本训练阶段最后一条经历上标记结束，以完成尾批。
"""
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from mematlas import MemAtlasConfig, Memory, MemoryEpisode, MemoryEvent, MemoryUnit
from . import AddOptions, MemRLConfig


async def run(
    tasks: Sequence[str],
    agent_call: Callable[[list[dict[str, str]]], Awaitable[str]],
    evaluate_answer: Callable[[str, str], Awaitable[bool]],
    *,
    providers: Mapping[str, Any],
) -> list[str]:
    """tasks 是一个训练阶段；Agent 与真实业务评价均由调用方提供。"""
    config = MemAtlasConfig(method=MemRLConfig(), providers=providers,
                            scope={"user_id": "alice"})
    answers = []
    async with Memory(config) as memory:
        for task_id, task in enumerate(tasks):
            events = [MemoryEvent("message", task, role="user")]
            used = await memory.search(task)
            # 全部条目交给 Agent，完整贡献保留原方法学习所需的查询关联。
            text = "\n".join(item["content"] for item in used.items)
            answer = await agent_call([
                {"role": "system", "content": f"相关记忆：\n{text}"},
                {"role": "user", "content": task},
            ])
            events.append(MemoryEvent("message", answer, role="assistant"))
            success = await evaluate_answer(task, answer)
            unit = MemoryUnit.from_episode(MemoryEpisode(
                task=task, output=answer, events=events,
                success=success, applied_memory=[used],
                extensions={"memrl": AddOptions(
                    task_id=task_id, end_of_training=task_id == len(tasks) - 1)},
            ))
            receipt = await memory.add(unit)
            await memory.episode_end(receipt.episode_ref)
            answers.append(answer)
        # 满批自动演化；最后一条 learn 内部完成尾批，无需额外触发接口。
    return answers
