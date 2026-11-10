"""核验注册 AWMQA 回放产物，并通过公共接口检查恢复、冻结检索和清理。

本工具不调用模型，不把回放视为真实浏览器或完整 Benchmark；保留任务实际失败。
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import re
from typing import Any, Sequence

from mematlas import MemAtlasConfig, Memory
from mematlas.core.files import atomic_write_json
from mematlas.core.plugin import MethodServices
from mematlas.core.provenance import json_sha256
from mematlas.methods.awm.core import AWMCore
from mematlas.test.awm.mind2web_smoke import _env_helpers


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def checked_state(path: Path) -> dict[str, Any]:
    """先验证公共快照校验和，再返回方法状态。"""
    payload = read_json(path)
    digest = payload.pop("payload_sha256")
    if json_sha256(payload) != digest:
        raise ValueError(f"Snapshot checksum mismatch: {path}")
    return payload["state"]


class NoModel:
    """恢复与整站文件检索不应调用模型；意外调用时立即失败。"""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, *args: Any, **kwargs: Any) -> str:
        self.calls += 1
        raise AssertionError("AWM evidence audit must not call a model")


async def check(run_dir: Path, config_path: Path, output: Path) -> dict[str, Any]:
    """核对现有回放并在新的输出目录保存审计产物，不修改原运行目录。"""
    summary = read_json(run_dir / "summary.json")
    manifest = read_json(run_dir / "manifest.json")
    state = checked_state(run_dir / "checkpoint-after-test.json")
    assert state == checked_state(run_dir / "checkpoint-before-test.json")
    assert state == read_json(run_dir / "final-memory.json")
    assert summary["live_browser"] is False
    assert len(summary["tasks"]) == 3

    traces = read_json(run_dir / "qa-traces.json")
    traces_by_id = {row["case_id"]: row for row in traces}
    episodes = [read_json(run_dir / filename) for filename in (
        "episode-train-1.json", "episode-train-2.json", "episode-test.json")]
    core = AWMCore("mind2web")
    tasks = []
    for index, (task, episode) in enumerate(zip(summary["tasks"], episodes)):
        trace = traces_by_id[task["case_id"]]
        actions = [event["content"] for event in episode["events"] if event["kind"] == "action"]
        assert actions == task["predicted_actions"]
        assert actions == [call["action"] for call in trace["model_calls"]]
        for action, call in zip(actions, trace["model_calls"]):
            blocks = re.findall(r"```(?:python\s*\n)?(.*?)```", call["raw_output"], re.S)
            assert action == (blocks[-1].strip() if blocks else call["raw_output"].strip())
        # 按原生动作解析器重算回放成败，避免只对比两个保存的成败字段。
        matched = 0
        for action, target in zip(actions, task["target_actions"]):
            if _env_helpers().parse_act_str(action) != _env_helpers().parse_act_str(target):
                break
            matched += 1
        assert episode["success"] is (matched == len(task["target_actions"]))
        assert episode["success"] is task["environment_success"]
        if not episode["success"]:
            assert actions != task["target_actions"]
        if index < 2:
            saved = state["episodes"][episode["episode_id"]]
            assert saved["success"] is episode["success"]
            assert [event["content"] for event in saved["events"]] == [
                event["content"] for event in episode["events"]]
        else:
            assert episode["episode_id"] not in state["episodes"]
        tasks.append({key: task[key] for key in (
            "split", "case_id", "environment_success", "predicted_actions", "target_actions")})

    # 归纳后的文件必须来自保存的实际输出，来源必须来自选中的训练经历。
    for history in state["induction_history"]:
        assert core.filter_response(history["output"], history["website"]) == history["filtered"]
        sources = list(dict.fromkeys(event["event_id"]
            for identity in history["selected_episode_ids"] for event in state["episodes"][identity]["events"]))
        assert history["source_event_ids"] == sources
    assert len(state["episodes"]) == 2
    assert len(state["induction_history"]) == 2
    assert summary["workflow_changed_after_second_training_task"] is True
    assert summary["second_task_retrieved_updated_workflow"] is True
    assert summary["test_retrieved_workflow"] is True
    for task in summary["tasks"][1:]:
        assert all(contents == [task["workflow_before"]] for contents in task["retrieved_contents"])

    output.mkdir(parents=True, exist_ok=False)
    model = NoModel()
    # 公共恢复签名包含 providers；保留原模型配置，注入 NoModel 阻止任何实际调用。
    config = MemAtlasConfig.load(config_path)
    config = replace(config, runtime=replace(config.runtime, artifact_dir=str(output / "runtime")))
    async with Memory(config, services=MethodServices(llm=model)) as memory:
        await memory.load(run_dir / "checkpoint-after-test.json")
        await memory.save(output / "restored.json")
        assert checked_state(output / "restored.json") == state
        async with memory.read_only():
            used = await memory.search("Inspect the saved workflow", extensions={
                "awm": {"website": manifest["website"]}})
        assert [item["content"] for item in used.items] == [state["workflow_text"][manifest["website"]]]
        await memory.save(output / "after-frozen.json")
        assert checked_state(output / "after-frozen.json") == state
        atomic_write_json(output / "retrieval.json", used.to_dict())
        await memory.reset()
        assert not (await memory.search("Inspect reset", extensions={
            "awm": {"website": manifest["website"]}})).items
        await memory.save(output / "after-reset.json")
        cleared = checked_state(output / "after-reset.json")
        assert not any(cleared[key] for key in (
            "episodes", "episode_order", "pending", "workflow_text", "workflow_sources", "induction_history"))

    assert model.calls == 0
    atomic_write_json(output / "memory-readable.json", deepcopy(state))
    report = {"kind": "registered AWMQA official-data replay audit", "live_browser": False,
        "tasks": tasks, "training_episodes": len(state["episodes"]),
        "induction_count": len(state["induction_history"]), "workflow_files": len(state["workflow_text"]),
        "workflow_updated": True, "retrieved_updated_workflow": True, "test_not_learned": True,
        "frozen_state_equal": True, "public_restore_equal": True, "reset_empty": True,
        "model_calls": model.calls, "quality_claim": False}
    atomic_write_json(output / "checks.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="已完成的 registered_qa_smoke 输出目录")
    parser.add_argument("--config", type=Path, required=True, help="原运行的公共配置，保留 providers 以匹配快照签名")
    parser.add_argument("--output", type=Path, required=True, help="新的审计目录，不覆盖已有产物")
    args = parser.parse_args(argv)
    print(json.dumps(asyncio.run(check(args.run_dir, args.config, args.output)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
