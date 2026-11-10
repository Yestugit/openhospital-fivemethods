"""Run registered AWMQA on a bounded, action-checked Mind2Web replay.

The replay exposes official ranked observations and advances only when the
Agent emits the annotated action. It is not a live browser or full benchmark.
"""
from __future__ import annotations
from mematlas.evaluation.validation import submit_episode

import argparse
import asyncio
import json
import os
from pathlib import Path
import pickle
from typing import Any

from mematlas import ExperimentConfig, MemAtlasConfig, Memory
from mematlas.core.evaluation import BenchmarkCase
from mematlas.core.files import atomic_write_json
from mematlas.core.plugin import MethodServices
from mematlas.core.provenance import file_sha256
from mematlas.core.registry import default_registry
from mematlas.evaluation.qa import _qa_type
from mematlas.providers import OpenAICompatibleLLM

from mematlas.test.awm.mind2web_smoke import _env_helpers, _load_env, _read_json, _test_rows


class Mind2WebReplay:
    def __init__(self, row: dict[str, Any], top_k: int):
        self.row = row
        self.index = 0
        env = _env_helpers()
        self.targets = [env.get_target_obs_and_act(step)[1] for step in row["actions"]]
        self.observations = [env.get_top_k_obs(step, top_k, use_raw=True)[0]
                             for step in row["actions"]]
        self.actions: list[str] = []
        self.success: bool | None = None

    async def reset(self):
        self.index = 0
        self.success = None
        return {"observation": self.observations[0], "done": False}

    async def step(self, action: str):
        self.actions.append(action)
        env = _env_helpers()
        try:
            actual = env.parse_act_str(action)
            expected = env.parse_act_str(self.targets[self.index])
            accepted = actual == expected
        except Exception:
            accepted = False
        if not accepted:
            self.success = False
            return {"observation": "The attempted action did not advance this task.",
                    "done": True, "success": False}
        self.index += 1
        if self.index == len(self.targets):
            self.success = True
            return {"observation": "Task completed.", "done": True, "success": True}
        return {"observation": self.observations[self.index], "done": False}

    async def close(self):
        return None


def choose_tasks(train_rows, test_rows):
    train_sites = {row["website"] for row in train_rows}
    common = sorted(train_sites & {row["website"] for row in test_rows})
    sites = [site for site in common if sum(row["website"] == site for row in train_rows) >= 2]
    if not sites:
        raise ValueError("Mind2Web needs a website with two train tasks and one test task")
    website = "sports.yahoo" if "sports.yahoo" in sites else sites[0]
    training = sorted((row for row in train_rows if row["website"] == website),
                      key=lambda row: (len(row["actions"]), row["annotation_id"]))[:2]
    testing = min((row for row in test_rows if row["website"] == website),
                  key=lambda row: (len(row["actions"]), row["annotation_id"]))
    return website, training, testing


def add_candidate_scores(rows, candidate_results):
    scores, ranks = candidate_results["scores"], candidate_results["ranks"]
    for row in rows:
        for step in row["actions"]:
            identity = f"{row['annotation_id']}_{step['action_uid']}"
            for candidates in (step["pos_candidates"], step["neg_candidates"]):
                for candidate in candidates:
                    key = candidate["backend_node_id"]
                    candidate["score"] = scores[identity][key]
                    candidate["rank"] = ranks[identity][key]


def make_case(row, scope):
    return BenchmarkCase(
        case_id=str(row["annotation_id"]),
        query=row["confirmed_task"],
        scope=scope,
        metadata={
            "awm": {"website": row["website"], "domain": row["domain"],
                    "subdomain": row["subdomain"], "max_steps": len(row["actions"])},
            "memory_extensions": {"awm": {"website": row["website"]}},
        },
    )


async def save_method_state(memory, path):
    """Persist through Memory's public API and return only algorithm state."""
    await memory.save(path)
    return json.loads(path.read_text(encoding="utf-8"))["state"]


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.env_file:
        _load_env(args.env_file)
    if not os.getenv("AMEM_API_KEY"):
        raise ValueError("Set AMEM_API_KEY in .env.local or the environment")
    train_rows = _read_json(args.train_json)
    test_rows = _test_rows(args.test_zip)
    with args.scores.open("rb") as handle:
        candidate_results = pickle.load(handle)
    website, training, testing = choose_tasks(train_rows, test_rows)
    selected = [*training, testing]
    add_candidate_scores(selected, candidate_results)
    rows_by_task = {row["confirmed_task"]: row for row in selected}
    if len(rows_by_task) != len(selected):
        raise ValueError("Selected task texts are not unique; cannot isolate replay environments")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    scope = {"user_id": "awm-registered-qa-smoke"}
    model_name = os.getenv("AMEM_LLM_MODEL", "qwen-flash")
    base_url = os.getenv("AMEM_BASE_URL")
    config = MemAtlasConfig.from_dict({
        "method": {"method_id": "awm", "params": {"profile": "mind2web", "induction_mode": "online"}},
        "scope": scope,
        "runtime": {"artifact_dir": str(output / "runtime")},
        "providers": {"llm": {"model": model_name, "base_url": base_url,
                      "api_key_env": "AMEM_API_KEY",
                      "options": {"extra_body": {"enable_thinking": False}}}},
    })
    atomic_write_json(output / "memory-config.json", config.to_dict())
    services = MethodServices(artifact_dir=output / "services")
    services.llm = OpenAICompatibleLLM(model=model_name, base_url=base_url,
        api_key_env="AMEM_API_KEY", default_options={"extra_body": {"enable_thinking": False}},
        usage=services.usage)
    replays: dict[str, Mind2WebReplay] = {}
    qa_traces: list[dict[str, Any]] = []
    services.objects["awm_environment_factory"] = lambda task, _options: replays.setdefault(
        task, Mind2WebReplay(rows_by_task[task], args.top_k))
    services.objects["awm_trace_sink"] = qa_traces.append
    experiment = ExperimentConfig(methods=(config,), benchmark=None, protocol="experience_train_test")
    plugin = default_registry().get("awm")
    qa_class = _qa_type(plugin)
    qa_class.validate_cases(experiment, (), tuple(make_case(row, scope) for row in selected))
    service = {"registered_qa": f"{qa_class.__module__}.{qa_class.__qualname__}",
               "benchmark": "Mind2Web official data replay", "live_browser": False,
               "website": website, "model": model_name, "top_k": args.top_k}
    atomic_write_json(output / "manifest.json", {
        **service,
        "protocol": "two real-Qwen training tasks via registered AWMQA, then one frozen test task",
        "quality_claim": False,
        "sources": {"upstream_revision": "8c0ff8cd11d648c8fceb99e4e42f37e3b75381b1",
            "train_sha256": file_sha256(args.train_json), "test_sha256": file_sha256(args.test_zip),
            "scores_sha256": file_sha256(args.scores)},
        "tasks": [{"case_id": row["annotation_id"], "task": row["confirmed_task"],
                   "action_count": len(row["actions"]), "split": "train" if row in training else "test"}
                  for row in selected],
    })
    results = []
    async with Memory(config, services=services) as memory:
        services.objects["answer_llm"] = memory.services.llm
        qa = qa_class(experiment, services)
        try:
            for index, row in enumerate(training, start=1):
                before = await save_method_state(memory, output / f"checkpoint-before-train-{index}.json")
                case = make_case(row, scope)
                episode = await qa(case, memory)
                await submit_episode(memory, episode)
                after = await save_method_state(memory, output / f"checkpoint-after-train-{index}.json")
                atomic_write_json(output / f"episode-train-{index}.json", episode.to_dict())
                result = {"split": "train", "case_id": case.case_id,
                    "task": row["confirmed_task"], "target_actions": [
                        _env_helpers().get_target_obs_and_act(step)[1] for step in row["actions"]],
                    "predicted_actions": replays[row["confirmed_task"]].actions,
                    "environment_success": replays[row["confirmed_task"]].success,
                    "workflow_before": before["workflow_text"].get(website, ""),
                    "workflow_after": after["workflow_text"].get(website, ""),
                    "induction_count": len(after["induction_history"]),
                    "retrieved_contents": [[item["content"] for item in contribution.items]
                        for contribution in episode.applied_memory]}
                results.append(result)
                atomic_write_json(output / f"task-train-{index}.json", result)

            before_test = await save_method_state(memory, output / "checkpoint-before-test.json")
            case = make_case(testing, scope)
            async with memory.read_only():
                episode = await qa(case, memory)
            after_test = await save_method_state(memory, output / "checkpoint-after-test.json")
            atomic_write_json(output / "episode-test.json", episode.to_dict())
            test_result = {"split": "test", "case_id": case.case_id,
                "task": testing["confirmed_task"], "target_actions": [
                    _env_helpers().get_target_obs_and_act(step)[1] for step in testing["actions"]],
                "predicted_actions": replays[testing["confirmed_task"]].actions,
                "environment_success": replays[testing["confirmed_task"]].success,
                "workflow_before": before_test["workflow_text"].get(website, ""),
                "workflow_after": after_test["workflow_text"].get(website, ""),
                "memory_unchanged_during_test": before_test == after_test,
                "retrieved_contents": [[item["content"] for item in contribution.items]
                    for contribution in episode.applied_memory]}
            results.append(test_result)
            atomic_write_json(output / "task-test.json", test_result)
            atomic_write_json(output / "qa-traces.json", qa_traces)
            atomic_write_json(output / "final-memory.json", after_test)
            atomic_write_json(output / "usage.json", memory.stats())
        finally:
            await qa.close()

    summary = {**service, "tasks": results,
        "workflow_changed_after_second_training_task": results[1]["workflow_before"] != results[1]["workflow_after"],
        "second_training_task_retrieved_prior_workflow": bool(results[1]["retrieved_contents"] and
            results[1]["retrieved_contents"][0]),
        "test_retrieved_latest_workflow": bool(results[2]["retrieved_contents"] and
            results[2]["retrieved_contents"][0] and
            results[2]["retrieved_contents"][0][0] == results[1]["workflow_after"]),
        "test_memory_unchanged": results[2]["memory_unchanged_during_test"],
        "quality_claim": False}
    atomic_write_json(output / "summary.json", summary)
    return summary


def parser():
    root = Path(__file__).resolve().parents[2]
    data = root / ".mematlas" / "awm-validation" / "mind2web-data"
    result = argparse.ArgumentParser(description="Two-task registered AWMQA Mind2Web replay smoke")
    result.add_argument("--train-json", type=Path, default=data / "train_10.json")
    result.add_argument("--test-zip", type=Path, default=data / "test.zip")
    result.add_argument("--scores", type=Path, default=data / "scores_all_data.pkl")
    result.add_argument("--output", type=Path, default=data / "registered-qa-20261103")
    result.add_argument("--env-file", type=Path, default=root / ".env.local")
    result.add_argument("--top-k", type=int, default=5)
    return result


if __name__ == "__main__":
    report = asyncio.run(run(parser().parse_args()))
    print(json.dumps({key: report[key] for key in (
        "registered_qa", "website", "workflow_changed_after_second_training_task",
        "second_training_task_retrieved_prior_workflow", "test_retrieved_latest_workflow",
        "test_memory_unchanged")},
        ensure_ascii=False, indent=2))
