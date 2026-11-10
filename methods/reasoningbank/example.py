"""Minimal Memory v2 flow; the success label must come from the external Agent."""
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
import tempfile
from typing import Any

from mematlas import MemAtlasConfig, Memory, MemoryEpisode, MemoryEvent, MemoryUnit
from . import ReasoningBankConfig


async def run(
    agent_call: Callable[[list[dict[str, str]]], Awaitable[str]],
    external_feedback: Callable[[str, str], Awaitable[tuple[bool, str | None]]],
    *,
    providers: Mapping[str, Any],
) -> None:
    config = MemAtlasConfig(
        method=ReasoningBankConfig(implementation_mode="standard", top_k=2),
        providers=providers,
        scope={"user_id": "alice"},
    )
    async with Memory(config) as memory:
        task = "Find the requested item in the account portal"
        start = MemoryEvent("message", task, role="user")
        contribution = await memory.search(task, top_k=2)
        prompt = "\n".join(item["content"] for item in contribution.items)
        answer = await agent_call([{"role": "system", "content": prompt}, {"role": "user", "content": task}])
        finish = MemoryEvent("message", answer, role="assistant")
        success, feedback = await external_feedback(task, answer)
        episode = MemoryEpisode(task=task, events=[start, finish], output=answer,
                                success=success, feedback=feedback, applied_memory=[contribution])
        unit = MemoryUnit.from_episode(episode)
        receipt = await memory.add(unit)
        await memory.episode_end(receipt.episode_ref)
        checkpoint = Path(tempfile.mkdtemp()) / "reasoningbank-checkpoint.json"
        await memory.save(checkpoint)
        after_learning = await memory.search(task, top_k=2)
        print({"items": contribution.items, "after_learning": after_learning.items,
               "checkpoint": str(checkpoint)})
