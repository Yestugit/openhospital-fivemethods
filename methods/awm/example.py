"""给已有异步 Agent 接入 AWM；配置与数据要求查看 help(mematlas.methods.awm)。

直接接入已有 Agent 时，调用 :func:`run_agent`，它按固定顺序完成：

    retrieve -> Agent 执行 -> observe 每个事件 -> learn 完整 episode

网页测评使用注册的 ``AWMQA``，需要把外部环境工厂放入服务对象：

    services = MethodServices(objects={"awm_environment_factory": factory})
    configure_environment_factory(services, factory)
    await ExperimentRunner().evaluate(experiment_config, services={"awm": services})

普通 Agent 可直接使用 ``run``。其 ``agent_call(task, contribution)`` 返回真实轨迹，
例如 Mind2Web 的 ``MemoryEpisode(task=..., events=[observation, action, ...],
success=实际环境结果, applied_memory=[contribution])``；真实领域和子领域由任务
映射提供。随后 ``run`` 负责创建 Memory 并在每个任务的合适位置调用公共接口。

工厂接收 ``(task, public_options)``，返回实现 ``reset``、``step(action)`` 和
``close`` 的环境；状态必须包含 ``observation``、``done``，终止时还要有真实
``success``。命令行导入已完成的真实轨迹并保存状态。参数查看
``python -m mematlas.methods.awm.example --help``。
"""
from __future__ import annotations
import argparse
import asyncio
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, Callable, Awaitable

from mematlas import MemAtlasConfig, Memory, MemoryEpisode, MemoryUnit
from .config import AWMConfig
from mematlas.core.models import MemoryContribution
from mematlas.core.files import atomic_write_json
from mematlas.core.plugin import MethodServices


def configure_environment_factory(
    services: MethodServices,
    factory: Callable[[str, Mapping[str, Any]], Any],
) -> MethodServices:
    """Register the external browser environment used by :class:`AWMQA`.

    The factory receives the task and public profile/site metadata, then returns
    an object implementing ``reset()``, ``step(action)`` and ``close()``. It must
    not expose reference answers or target actions to the model.
    """
    if not callable(factory):
        raise TypeError("factory must be callable")
    services.objects["awm_environment_factory"] = factory
    return services


async def run_agent(
    memory: Memory,
    task: str,
    website: str,
    agent: Callable[[str, MemoryContribution], Awaitable[MemoryEpisode]],
    *,
    domain: str | None = None,
    subdomain: str | None = None,
    end_of_training: bool = False,
) -> MemoryEpisode:
    """检索目标网站工作流，交给已有 Agent 执行，再记录事件并提交学习。

    Args:
        memory: 已打开的公共 Memory，使用 AWM 配置。
        task: 当前任务文本。
        website: 当前任务的实际网站标识，用于选择工作流文件。
        agent: 异步 Agent，接收任务与完整贡献，返回包含实际事件、真实反馈和
            必需 awm 字段的 MemoryEpisode；applied_memory 只记录实际使用的贡献。
        domain: Mind2Web 的真实领域标签；medicine/WebArena 不需要。
        subdomain: Mind2Web 的真实子领域标签；medicine/WebArena 不需要。
        end_of_training: 是否为当前 scope 的最后一份训练经历，用于处理剩余批次。

    Returns:
        Agent 返回且已提交 learn 的完整经历。工具执行和成败评价由 Agent/环境负责。
    """
    if not website.strip():
        raise ValueError("website must be the actual target website")
    contribution = await memory.search(task, extensions={"awm": {"website": website}})
    episode = await agent(task, contribution)
    if not isinstance(episode, MemoryEpisode) or not episode.events:
        raise ValueError("AWM external Agent must return a complete real MemoryEpisode")

    # The helper already knows the selected website; preserve other method data
    # returned by the Agent while ensuring learn groups this episode correctly.
    awm_options = dict(episode.extensions.get("awm", {}))
    awm_options.setdefault("website", website)
    if domain is not None:
        awm_options.setdefault("domain", domain)
    if subdomain is not None:
        awm_options.setdefault("subdomain", subdomain)
    if end_of_training:
        awm_options["end_of_training"] = True
    episode = replace(episode, extensions={**episode.extensions, "awm": awm_options})
    unit = MemoryUnit.from_episode(episode)
    receipt = await memory.add(unit)
    await memory.episode_end(receipt.episode_ref)
    return episode


async def run(
    tasks: Sequence[Mapping[str, str]],
    agent_call: Callable[[str, MemoryContribution], Awaitable[MemoryEpisode]],
    *,
    providers: Mapping[str, Any],
    scope: Mapping[str, str],
    profile: str = "mind2web",
) -> list[MemoryEpisode]:
    """Open AWM and apply the standard Agent-memory loop to real web tasks.

    Each task mapping contains ``task`` and ``website``. Mind2Web tasks also
    contain the real ``domain`` and ``subdomain`` labels. ``agent_call`` remains
    the user's Agent; it returns actual observation/action events, actual success
    feedback when available, and complete ``applied_memory`` for contributions
    it really used. ``providers`` contains the configured memory-model services.
    """
    config = MemAtlasConfig(
        method=AWMConfig(profile=profile),
        providers=providers,
        scope=scope,
    )
    episodes = []
    async with Memory(config) as memory:
        for index, row in enumerate(tasks):
            task = row["task"]
            episode = await run_agent(
                memory,
                task,
                row["website"],
                agent_call,
                domain=row.get("domain"),
                subdomain=row.get("subdomain"),
                end_of_training=index == len(tasks) - 1,
            )
            episodes.append(episode)
    return episodes


async def ingest(config:MemAtlasConfig,episodes:Sequence[MemoryEpisode],output:Path,
                 *,services:MethodServices|None=None)->None:
    """导入已完成的轨迹，保存每次学习前后及冻结检索后的状态。

    Args:
        config: AWM 公共配置，包含任务流程和模型服务。
        episodes: 完整 MemoryEpisode 序列，包含相应流程需要的真实反馈或标注轨迹。
        output: 尚不存在的运行目录，用于保存配置、快照、检索结果与统计。
        services: 可选的已有模型服务；省略时使用 config.providers。

    每个归属的最后一份经历自动补 end_of_training=True，由 learn 完成剩余归纳。
    本函数只导入经历，任务执行和评分应在提交前完成；save/close 不触发归纳。
    """
    if config.method.method_id != "awm":
        raise ValueError("This example requires awm")
    if not episodes:
        raise ValueError("Provide at least one completed trajectory")
    report = Memory.check(config, services=services)
    if not report["ready"]:
        raise ValueError(f"AWM not ready: {report}")
    output.mkdir(parents=True, exist_ok=False)
    config = replace(config, runtime=replace(config.runtime, artifact_dir=str(output / "runtime")))
    atomic_write_json(output / "config.json", config.to_dict())
    last = {
        json.dumps(episode.scope or config.scope, sort_keys=True): index
        for index, episode in enumerate(episodes)
    }
    async with Memory(config, services=services) as memory:
        for index,episode in enumerate(episodes):
            await memory.save(output / f"before-{index + 1}.json")
            scope_key = json.dumps(episode.scope or config.scope, sort_keys=True)
            if index == last[scope_key]:
                options = {
                    **episode.extensions.get("awm", {}),
                    "end_of_training": True,
                }
                episode = replace(
                    episode,
                    extensions={**episode.extensions, "awm": options},
                )
            if not episode.events:
                raise ValueError("Example imports complete events, not unresolved episode IDs")
            unit = MemoryUnit.from_episode(episode)
            receipt = await memory.add(unit)
            await memory.episode_end(receipt.episode_ref)
            await memory.save(output / f"after-{index + 1}.json")
        websites = sorted({
            episode.extensions.get("awm", {}).get("website", "medicine")
            for episode in episodes
        })
        contributions = {}
        async with memory.read_only():
            for website in websites:
                used = await memory.search(
                    "next task", extensions={"awm": {"website": website}}
                )
                contributions[website] = used.to_dict()
        await memory.save(output / "after-frozen.json")
        atomic_write_json(output / "retrieval.json", contributions)
        atomic_write_json(output / "stats.json", memory.stats())
    atomic_write_json(output / "run.json", {
        "kind": "completed-trajectory integration example",
        "benchmark_evaluation_executed": False,
        "quality_claim": False,
        "episodes": len(episodes),
        "websites": websites,
    })


def main(argv:Sequence[str]|None=None):
    parser = argparse.ArgumentParser(
        description="AWM completed-trajectory import/Agent integration; no benchmark execution"
    )
    parser.add_argument("--config", type=Path, required=True,
                        help="Public AWM config; providers.llm is required")
    parser.add_argument("--episodes", type=Path, required=True,
                        help="JSON array of actual completed MemoryEpisode.to_dict records")
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory, never overwrite previous evidence")
    parser.add_argument("--env-file", type=Path,
                        help="Optional local dotenv file, never recorded")
    args = parser.parse_args(argv)
    if args.env_file:
        if not args.env_file.is_file():
            parser.error("--env-file must exist")
        from dotenv import load_dotenv
        load_dotenv(args.env_file,override=True)
    values = json.loads(args.episodes.read_text(encoding="utf-8"))
    if not isinstance(values, list):
        parser.error("--episodes must contain a JSON array")
    asyncio.run(ingest(
        MemAtlasConfig.load(args.config),
        [MemoryEpisode.from_dict(value) for value in values],
        args.output,
    ))


if __name__=='__main__':main()
