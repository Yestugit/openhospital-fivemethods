"""完整历史逐事件导入、保存检查点，再冻结测评 LoCoMo 前三题。

复用公共 ExperimentRunner，不修改 QA、Memory 或模型算法。
默认 event 写入模式下，事件的内容、顺序、时间和归属与普通历史导入相同。
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

from mematlas import ExperimentConfig, ExperimentRunner, MemAtlasConfig, MemoryEpisode
from mematlas.benchmarks.datasets import InMemoryBenchmark
from mematlas.benchmarks.locomo import LoCoMoBenchmark
from mematlas.core.evaluation import BenchmarkCase
from mematlas.core.models import content_text
from mematlas.core.provider_config import resolve_providers
from mematlas.methods.a_mem.example import _config_from_environment


def make_experiment(memory: MemAtlasConfig, data: Path, output: Path, *, test_limit: int = 3) -> ExperimentConfig:
    """每条原始历史是一份已完成输入经历，不使用任何测试答案生成训练数据。"""
    if memory.method.method_id != "a-mem" or memory.method.params.get("write_policy", "event") != "event":
        raise ValueError("Checkpointed history validation requires a-mem with write_policy=event")
    benchmark = LoCoMoBenchmark(data, max_conversations=1)
    tests = list(benchmark.cases())[:test_limit]
    if len(tests) != test_limit or not tests[0].history:
        raise ValueError("The first conversation must contain the requested questions and history")
    if any(case.scope != tests[0].scope or case.history != tests[0].history for case in tests):
        raise ValueError("All selected questions must share exactly the same complete history and scope")
    train = InMemoryBenchmark("locomo-history-events", [
        BenchmarkCase(case_id=f"history:{index}:{event.event_id}", query=content_text(event.content),
            scope=tests[0].scope,
            episode=MemoryEpisode(episode_id=f"history:{index}:{event.event_id}", events=[event]),
            metadata={"split": "train"})
        for index, event in enumerate(tests[0].history)
    ])
    reader = resolve_providers(memory.method.params, memory.providers)["llm"]
    reader = replace(reader, options={key: value for key, value in reader.options.items() if key != "response_format"})
    return ExperimentConfig(methods=(memory,), benchmark=benchmark, train_benchmark=train,
        protocol="experience_train_test", train_limit=len(tests[0].history), test_limit=test_limit,
        answer_model=reader if reader.model else None, output_dir=str(output))


async def run(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv
    load_dotenv(args.env_file, override=True)
    memory = _config_from_environment()
    experiment = make_experiment(memory, args.data, args.output, test_limit=args.test_limit)
    print(json.dumps({"stage": "preflight", "history_events": experiment.train_limit,
                      "test_questions": experiment.test_limit, "resume_from": str(args.resume_from) if args.resume_from else None}), flush=True)
    runner = ExperimentRunner()
    report = runner.check(experiment)
    if not report["ready"]:
        raise RuntimeError(f"Experiment is not ready: {report}")
    result = (await runner.evaluate(experiment, resume_from=args.resume_from))[0]
    print("Metrics:", dict(result.metrics), flush=True)
    print("Failures:", list(result.failures), flush=True)
    print("Output:", args.output.resolve(), flush=True)
    if result.failures or len(result.predictions) != args.test_limit:
        raise RuntimeError("Validation incomplete; inspect results and resume into a new output directory")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="新的输出目录")
    parser.add_argument("--resume-from", type=Path, help="上一轮实验输出根目录，使用框架验证过的检查点恢复")
    parser.add_argument("--test-limit", type=int, default=3)
    args = parser.parse_args(argv)
    if args.test_limit < 1:
        parser.error("--test-limit must be positive")
    if not args.data.is_file() or not args.env_file.is_file():
        parser.error("--data and --env-file must exist")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
