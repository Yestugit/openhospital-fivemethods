"""最小公共接口示例；导入本文件不会执行模型调用。先安装方法专用requirements.lock。"""
from mematlas import MemAtlasConfig, Memory, MemoryEvent, MemoryUnit
from mematlas.core.config import RuntimeConfig
from . import MemMachineConfig


async def run(agent_call, *, providers, python_executable, feedback):
    """传入现有Agent、模型配置、独立环境解释器和真实评价；调用run才执行。"""
    config = MemAtlasConfig(method=MemMachineConfig(embedding_dims=1024),
        providers=providers, scope={"user_id": "doctor-demo", "agent_id": "doctor"},
        runtime=RuntimeConfig(executor="subprocess", python_executable=python_executable))
    async with Memory(config) as memory:
        views = await memory.index()
        task = "患者描述了用药后的不适，请整理下一步问诊要点。"
        used = await memory.search(task)
        answer = await agent_call(task, used)
        events = [MemoryEvent("message", task, role="user"),
                  MemoryEvent("message", answer, role="assistant")]
        accepted = await memory.add(MemoryUnit(events=events))
        # OpenHospital先结束Agent任务，稍后再提交评价及完整轨迹。
        await memory.episode_end(accepted.episode_ref)
        await memory.add(MemoryUnit(type="experience", events=events, task=task,
            output=answer, feedback=feedback, applied_memory=[used],
            related_refs=[accepted.write_ref]), episode_ref=accepted.episode_ref)
        return answer, views
