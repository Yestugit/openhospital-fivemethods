"""Minimal public workflow. No model selection or experiment protocol is imposed."""
from mematlas import MemAtlasConfig, Memory, MemoryEpisode, MemoryEvent, MemoryUnit, RuntimeConfig
from . import EMemConfig


async def run(agent_call, *, providers, source_root, python_executable, artifact_dir):
    config = MemAtlasConfig(method=EMemConfig(), providers=providers, scope={"agent_id": "doctor"},
        runtime=RuntimeConfig(executor="subprocess", python_executable=python_executable,
            source_roots={"emem": source_root}, artifact_dir=artifact_dir))
    async with Memory(config) as memory:
        await memory.index()
        task = "Use the actual current patient observations to decide the next step."
        used = await memory.search(task)
        answer = await agent_call(task, used)
        episode = MemoryEpisode(task=task, events=[MemoryEvent("message", task, role="user"),
            MemoryEvent("message", answer, role="assistant")], output=answer, applied_memory=[used])
        receipt = await memory.add(MemoryUnit.from_episode(episode))
        await memory.episode_end(receipt.episode_ref)
        return answer
