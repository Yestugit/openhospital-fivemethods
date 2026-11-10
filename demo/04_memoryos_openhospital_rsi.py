"""用 OpenHospital 检查 MemoryOS 的连续学习与冻结测评流程。

在外层项目根目录安装：
    python -m pip install -e ".[openai,openhospital]" faiss-cpu -r src/mematlas/integrations/openhospital_runtime/requirements.txt
配置 DASHSCOPE_API_KEY，然后运行：
    python src/mematlas/demo/04_memoryos_openhospital_rsi.py
心内科 1 个训练患者重复 2 轮；固定 1 个测试患者在 0/1/2 检查点测评，共 5 次诊疗。
小样本用于流程验收；MemoryOS 按自身阈值自动整合，结果不代表 RSI 性能结论。
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import socket
from uuid import uuid4

from mematlas import ExperimentConfig, ExperimentRunner, MemAtlasConfig, MemoryEvent, RuntimeConfig
from mematlas.core.registry import PluginRegistry
from mematlas.evaluation.qa import DefaultQA
from mematlas.integrations.openhospital import OpenHospitalServices
from mematlas.integrations.openhospital.config import SERVICE_URLS
from mematlas.methods.memoryos import MemoryOSConfig
from mematlas.methods.memoryos.method import MemoryOSPlugin


ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")


class MemoryOSHospitalQA(DefaultQA):
    @staticmethod
    def prepare_episode(case, episode, *, last_for_scope):
        # MemoryOS 接收 message：将真实诊疗轨迹与结果整理成一对消息，保留事件来源。
        sources = [event.event_id for event in episode.events]
        trajectory = json.dumps([event.to_dict() for event in episode.events], ensure_ascii=False)
        outcome = json.dumps({"output": episode.output, "feedback": episode.feedback,
                              "success": episode.success, "reward": episode.reward}, ensure_ascii=False)
        events = [MemoryEvent("message", text, role=role, episode_id=episode.episode_id,
                              event_id=f"{episode.episode_id}:{part}", scope=episode.scope,
                              timestamp=episode.events[0].timestamp,
                              extensions={"source_event_ids": sources})
                  for part, role, text in (("trajectory", "user", trajectory), ("outcome", "assistant", outcome))]
        return replace(episode, events=events)


class HospitalMemoryOSPlugin(MemoryOSPlugin):
    qa_factory = f"{__name__}:MemoryOSHospitalQA"


def make_experiment():
    run_dir = ROOT / ".mematlas" / "demo-runs" / f"04-memoryos-hospital-{uuid4().hex[:8]}"
    llm = {"model": "qwen-plus", "base_url": None, "api_key_env": "DASHSCOPE_API_KEY"}
    memory = MemAtlasConfig(
        # 小样本演示参数：第二次训练时即可迁移首例并检查自动归纳。
        method=MemoryOSConfig(short_term_capacity=1, mid_term_heat_threshold=1.5),
        providers={"llm": llm, "embedding": {"model": "text-embedding-v4", "base_url": None,
                  "dimensions": 1024, "api_key_env": "DASHSCOPE_API_KEY"}},
        scope={"user_id": "hospital-demo", "agent_id": "doctor"},
        runtime=RuntimeConfig(artifact_dir=str(run_dir / "memory")),
    )
    experiment = ExperimentConfig(
        methods=(memory,), protocol="experience_learning_curve", checkpoints=(0, 1, 2), seed=0,
        benchmark={"kind": "openhospital", "condition": "repeated", "departments": ("心内科",),
                   "evaluation_departments": ("心内科",), "cases_per_block": 1, "blocks": 2,
                   "heldout_per_department": 1, "advanced_per_department": 0,
                   "world_dir": str(run_dir / "world")},
        output_dir=str(run_dir / "evaluation"),
    )
    registry = PluginRegistry()
    registry.register(HospitalMemoryOSPlugin())
    return experiment, ExperimentRunner(registry=registry)


async def main():
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise RuntimeError("请先设置环境变量 DASHSCOPE_API_KEY")
    os.environ.setdefault("MODEL_API_KEY", os.environ["DASHSCOPE_API_KEY"])
    experiment, runner = make_experiment()
    report = runner.check(experiment)
    if not report["ready"]:
        raise RuntimeError(f"测评配置未就绪：{report}")
    # 给本 demo 的两项本地服务分配空闲端口，避免与已有医院实验冲突。
    for split in ("train", "primary_test"):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            SERVICE_URLS[split] = f"http://127.0.0.1:{probe.getsockname()[1]}"
    async with OpenHospitalServices(experiment):
        result = (await runner.evaluate(experiment))[0]
    progress = result.run_metadata.get("progress", {})
    print("结果目录：", result.run_metadata.get("output_dir", experiment.output_dir))
    print("完成训练数：", progress.get("completed_exposures"), "已学习数：", progress.get("learned_exposures"))
    print("失败数：", len(result.failures), "训练失败数：", progress.get("failed_exposures"))
    for point in result.learning_curve:
        print("检查点：", point["checkpoint"], "测试集：", point["split"],
              "诊断准确率：", point.get("diagnosis_accuracy"), "临床得分：", point.get("overall_clinical_score"))
    points = [point["checkpoint"] for point in result.learning_curve if point["split"] == "primary_test"]
    if result.failures or points != [0, 1, 2] or progress.get("completed_exposures") != 2 or progress.get("learned_exposures") != 2:
        raise RuntimeError(f"医院测评未完整完成：{result.failures}; progress={progress}")


if __name__ == "__main__":
    asyncio.run(main())
