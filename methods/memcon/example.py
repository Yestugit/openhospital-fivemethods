"""Minimal MemCon v2 event flow."""
from mematlas import MemAtlasConfig, Memory, MemoryEvent, MemoryUnit
from . import MemConConfig


async def run() -> None:
    config = MemAtlasConfig(method=MemConConfig(), scope={"user_id": "demo"})
    async with Memory(config) as memory:
        start = MemoryEvent("episode_start", "Organize the room", extensions={"task": "Organize the room"})
        action = MemoryEvent("action", "Inspect the table", extensions={"action": "Inspect the table"}, episode_id="demo")
        observation = MemoryEvent("observation", "The table is clear", extensions={"done": False}, episode_id="demo")
        receipt = await memory.add(MemoryUnit(type="experience", episode_ref="demo",
            task="Organize the room", events=[start, action, observation]))
        contribution = await memory.search("Organize the room")
        await memory.add(MemoryUnit(type="feedback", episode_ref=receipt.episode_ref,
            feedback={"success": True, "text": "completed"},
            related_refs=(receipt.write_ref,), applied_memory=[contribution]))
        await memory.episode_end(receipt.episode_ref)
