"""用默认 MemoryOS 测评真实 LoCoMo 的前 2 道题。

在外层项目根目录安装：
    python -m pip install -e ".[openai,benchmarks]" faiss-cpu
配置 DASHSCOPE_API_KEY，然后运行：
    python src/mematlas/demo/02_memoryos_locomo.py
数据位置：agent-memory-benchmarks/LoCoMo/data/locomo10.json。
首次运行会处理这 2 道题关联的完整会话历史，耗时可能较长。
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from uuid import uuid4

from mematlas import ExperimentConfig, ExperimentRunner, MemAtlasConfig, RuntimeConfig
from mematlas.methods.memoryos import MemoryOSConfig


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "agent-memory-benchmarks" / "LoCoMo" / "data" / "locomo10.json"
os.environ.setdefault("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL = os.getenv("QWEN_MODEL", "qwen-plus")


def main():
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise RuntimeError("请先设置环境变量 DASHSCOPE_API_KEY")
    if not DATA.is_file():
        raise FileNotFoundError(f"LoCoMo data not found: {DATA}")
    run_dir = ROOT / ".mematlas" / "demo-runs" / f"02-memoryos-locomo-{uuid4().hex[:8]}"
    llm = {"model": MODEL, "base_url": None, "api_key_env": "DASHSCOPE_API_KEY"}
    memory = MemAtlasConfig(
        method=MemoryOSConfig(),
        providers={"llm": llm, "embedding": {"model": "text-embedding-v4", "base_url": None,
                                                  "dimensions": 1024, "api_key_env": "DASHSCOPE_API_KEY"}},
        scope={"user_id": "locomo"},
        runtime=RuntimeConfig(artifact_dir=str(run_dir / "memory")),
    )
    experiment = ExperimentConfig(
        methods=(memory,),
        benchmark={"kind": "locomo", "path": str(DATA)},
        answer_model=llm,
        test_limit=2,
        output_dir=str(run_dir / "evaluation"),
    )
    runner = ExperimentRunner()
    report = runner.check(experiment)
    if not report["ready"]:
        raise RuntimeError(f"测评配置未就绪：{report}")
    result = asyncio.run(runner.evaluate(experiment))[0]
    print("指标：", dict(result.metrics))
    print("失败数：", len(result.failures))
    print("结果目录：", result.run_metadata.get("output_dir", experiment.output_dir))
    for prediction in result.predictions:
        print(prediction.case_id, "->", prediction.output)
    if result.failures or len(result.predictions) != 2:
        raise RuntimeError("LoCoMo 前两题未全部完成")


if __name__ == "__main__":
    main()
