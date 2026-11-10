"""Minimal AEL v2 portfolio-memory flow."""
from mematlas import MemAtlasConfig, Memory, MemoryEpisode, MemoryEvent, MemoryUnit
from . import AELConfig


async def run() -> None:
    config = MemAtlasConfig(method=AELConfig(), scope={"user_id": "portfolio"})
    async with Memory(config) as memory:
        event = MemoryEvent("message", "Evaluate the portfolio", role="user")
        contribution = await memory.search("portfolio", extensions={"ael": {"task_type": "portfolio_allocation"}})
        unit = MemoryUnit.from_episode(MemoryEpisode(task="portfolio allocation", events=[event], reward=0.7, applied_memory=[contribution], extensions={"ael": {"task_type": "portfolio_allocation"}}))
        receipt = await memory.add(unit)
        await memory.episode_end(receipt.episode_ref)
