"""给已有异步 Agent 接入 A-Mem；配置与数据要求查看 help(mematlas.methods.a_mem)。

run 展示 observe/retrieve/learn 及状态保存；命令行也支持少量问答测评。
参数查看 python -m mematlas.methods.a_mem.example --help。
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
import os

from mematlas import ExperimentConfig, ExperimentRunner, MemAtlasConfig, Memory, MemoryEpisode, MemoryEvent, MemoryUnit
from mematlas.core.files import atomic_write_json
from mematlas.core.contributions import contribution_text
from mematlas.core.plugin import MethodServices
from mematlas.core.provider_config import resolve_providers
from mematlas.methods.a_mem import AMemConfig


def _config_from_environment() -> MemAtlasConfig:
    """构建演示配置；读取变量名，不在配置或日志中保存密钥内容。"""
    required = ("AMEM_BASE_URL", "AMEM_LLM_MODEL", "AMEM_EMBEDDING_MODEL")
    missing = [key for key in required if not os.getenv(key, "").strip()]
    if missing:
        raise ValueError("请在本地配置中填写：" + ", ".join(missing))
    dimensions = os.getenv("AMEM_EMBEDDING_DIMS", "").strip()
    if dimensions and (not dimensions.isdigit() or int(dimensions) < 1):
        raise ValueError("AMEM_EMBEDDING_DIMS 须为空或正整数")
    return MemAtlasConfig(method=AMemConfig(), scope={"user_id": "alice"},
        providers={
            "llm": {"model": os.environ["AMEM_LLM_MODEL"].strip(),
                    "base_url": os.environ["AMEM_BASE_URL"].strip(), "api_key_env": "AMEM_API_KEY",
                    "options": {"response_format": {"type": "json_object"},
                                "extra_body": {"enable_thinking": False}}},
            "embedding": {"model": os.environ["AMEM_EMBEDDING_MODEL"].strip(),
                          "base_url": os.getenv("AMEM_EMBEDDING_BASE_URL", "").strip() or os.environ["AMEM_BASE_URL"].strip(),
                          "api_key_env": "AMEM_EMBEDDING_API_KEY" if os.getenv("AMEM_EMBEDDING_API_KEY") else "AMEM_API_KEY",
                          "dimensions": int(dimensions) if dimensions else None},
        })


async def run(
    tasks: Sequence[str],
    agent_call: Callable[[list[dict[str, str]]], Awaitable[str]],
    *,
    config: MemAtlasConfig,
    output: Path,
    services: MethodServices | None = None,
) -> list[str]:
    """沿用已有 Agent，连续执行任务并保存每轮经历、完整贡献与记忆快照。

    Args:
        tasks: 按顺序交给 Agent 的任务文本。
        agent_call: 用户已有的异步 Agent，接收消息列表并返回回答文本。
        config: 使用 AMemConfig() 或 a-mem 公共字典配置的 MemAtlasConfig。
        output: 新建的运行目录，必须尚不存在，保存经历、快照与统计。
        services: 可选的已有模型服务；省略时从 config.providers 构建。

    Returns:
        与 tasks 顺序一致的回答列表。
    """
    if config.method.method_id != "a-mem":
        raise ValueError("This example requires method_id=a-mem")
    report = Memory.check(config, services=services)
    if not report["ready"]:
        raise RuntimeError(f"A-Mem configuration is not ready: {report}")
    output.mkdir(parents=True, exist_ok=False)
    answers, records = [], []
    async with Memory(config, services=services) as memory:
        for index, task in enumerate(tasks, start=1):
            events = [MemoryEvent("message", task, role="user")]
            used = await memory.search(task)
            text = contribution_text(used)
            messages = [{"role": "system", "content": f"相关记忆：\n{text}"},
                        {"role": "user", "content": task}]
            answer = await agent_call(messages)
            events.append(MemoryEvent("message", answer, role="assistant"))
            episode = MemoryEpisode(task=task, output=answer, events=events, applied_memory=[used])
            unit = MemoryUnit.from_episode(episode)
            receipt = await memory.add(unit)
            await memory.episode_end(receipt.episode_ref)
            await memory.save(output / f"memory-{index}.json")
            records.append({"messages": messages, "episode": episode.to_dict()})
            atomic_write_json(output / "agent-records.json", records)
            answers.append(answer)
        atomic_write_json(output / "stats.json", memory.stats())
        print("Memory operation records:", memory.services.artifact_dir)
    return answers


async def _main(args: argparse.Namespace) -> None:
    config = MemAtlasConfig.load(args.config) if args.config else _config_from_environment()
    if config.method.method_id != "a-mem" or config.method.params.get("implementation_mode") == "offline":
        raise ValueError("Real-model example requires standard a-mem configuration")
    llm_config = resolve_providers(config.method.params, config.providers)["llm"]
    # JSON 输出约束用于记忆分析；外部 Agent/QA 仍返回普通回答文本。
    llm_config = replace(llm_config, options={key: value for key, value in llm_config.options.items()
                                            if key != "response_format"})
    if args.benchmark:
        # 题数上限只限制问题数量，关联的完整会话历史仍保留。
        experiment = ExperimentConfig(
            methods=(config,), benchmark={"kind": args.benchmark, "path": str(args.data)},
            answer_model=llm_config, test_limit=args.test_limit, output_dir=str(args.output),
        )
        result = (await ExperimentRunner().evaluate(experiment))[0]
        print("Metrics:", dict(result.metrics))
        print("Failures:", len(result.failures))
        print("Output:", args.output.resolve())
        if result.failures or len(result.predictions) != args.test_limit:
            raise RuntimeError("Requested benchmark cases did not all complete; inspect saved results")
        return
    from mematlas.providers import OpenAICompatibleLLM
    agent = OpenAICompatibleLLM(model=llm_config.model, base_url=llm_config.base_url,
        api_key_env=llm_config.api_key_env, default_options=llm_config.options)
    try:
        tasks = args.task or ["请记住：Helios 服务的端口是 8443。", "Helios 服务的端口后来改为了 9443。",
                              "Helios 服务最新的端口是多少？"]
        for answer in await run(tasks, agent.complete, config=config, output=args.output):
            print(answer)
    finally:
        if getattr(agent, "_client", None) is not None:
            await agent._client.close()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="A-Mem 真实服务 Agent demo 或少量 Benchmark；保留运行证据")
    parser.add_argument("--config", type=Path, help="公共 Memory 配置；省略则使用 AMEM_* 环境变量")
    parser.add_argument("--env-file", type=Path, help="本地 .env 文件，仅加载到本进程；参考同目录 .env.example")
    parser.add_argument("--output", type=Path, required=True, help="新的结果目录，不覆盖已有运行")
    parser.add_argument("--task", action="append", help="Agent 任务，可重复；默认演示记录、更新事实与查询")
    parser.add_argument("--benchmark", choices=("locomo", "longmemeval"), help="改为运行 Benchmark")
    parser.add_argument("--data", type=Path, help="Benchmark 原始数据路径")
    parser.add_argument("--test-limit", type=int, help="题数上限，LoCoMo 默认 3，LongMemEval 默认 1；单归属运行")
    args = parser.parse_args(argv)
    if args.test_limit is None:
        args.test_limit = 1 if args.benchmark == "longmemeval" else 3
    if args.benchmark and (args.data is None or not args.data.is_file()):
        parser.error("--benchmark requires an existing --data file")
    if args.test_limit < 1:
        parser.error("--test-limit must be positive")
    if args.benchmark and args.task:
        parser.error("--task is for the Agent demo; omit it for benchmark execution")
    if args.env_file:
        if not args.env_file.is_file():
            parser.error("--env-file 必须是已存在的本地文件")
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=True)
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
