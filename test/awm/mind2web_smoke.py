"""One-train/one-step Mind2Web smoke run through the public AWM Memory API.

This is a bounded integration probe, not a benchmark result or paper reproduction.
It reads official task data and pinned upstream preprocessing helpers, learns one
canonical training trajectory, retrieves its workflow, then scores one test action.
"""
from __future__ import annotations
from mematlas.evaluation.validation import submit_episode

import argparse
import asyncio
import importlib.util
import json
from functools import lru_cache
import os
from pathlib import Path
import pickle
import re
from typing import Any
from zipfile import ZipFile

from mematlas import MemAtlasConfig, Memory, MemoryEpisode, MemoryEvent
from mematlas.core.contributions import contribution_text
from mematlas.core.provenance import file_sha256


@lru_cache(maxsize=1)
def _env_helpers():
    reference_root = Path(__file__).parent / "reference" / "mind2web"
    return _module("mind2web_env", reference_root / "utils" / "env.py")


def choose_short_pair(train: list[dict], test: list[dict]) -> tuple[dict, dict]:
    """Choose the shortest train/test tasks sharing a site, with stable ties."""
    train_by_site: dict[str, list[dict]] = {}
    test_by_site: dict[str, list[dict]] = {}
    for sample in train:
        train_by_site.setdefault(str(sample["website"]), []).append(sample)
    for sample in test:
        test_by_site.setdefault(str(sample["website"]), []).append(sample)
    common = sorted(set(train_by_site) & set(test_by_site))
    candidates = []
    for website in common:
        before = min(train_by_site[website], key=lambda row: (len(row["actions"]), str(row.get("annotation_id", ""))))
        after = min(test_by_site[website], key=lambda row: (len(row["actions"]), str(row.get("annotation_id", ""))))
        candidates.append((len(before["actions"]) + len(after["actions"]), website, before, after))
    if not candidates:
        raise ValueError("No Mind2Web website has both training and test trajectories")
    _, _, before, after = min(candidates, key=lambda row: (row[0], row[1]))
    return before, after


def score_action(response: str, target: str, positive_candidates: list[dict]) -> dict[str, float]:
    """Apply Mind2Web's upstream action parser and native one-step metrics."""
    env = _env_helpers()
    blocks = re.findall(r"```(?:[a-zA-Z0-9_+-]*\s*\n)?(.*?)```", response, re.S)
    action = blocks[-1].strip() if blocks else response.strip()
    pred_op, pred_id, pred_value = env.parse_act_str(action)
    target_op, _, target_value = env.parse_act_str(target)
    positive_ids = {str(row["backend_node_id"]) for row in positive_candidates[:1]}
    return {
        "element_acc": float(pred_id in positive_ids),
        "action_f1": float(env.calculate_f1(env.construct_act_str(pred_op, pred_value),
                                              env.construct_act_str(target_op, target_value))),
        "step_success": float(action == target),
    }


def _load_env(path: Path) -> None:
    """Load local model settings without printing or storing secret values."""
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() and not key.lstrip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pinned Mind2Web helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return value


def _test_rows(path: Path) -> list[dict]:
    rows = []
    with ZipFile(path) as archive:
        names = sorted(name for name in archive.namelist()
                       if name.lower().endswith(".json") and "test_task" in name.lower())
        for name in names:
            # The official test archive is intentionally password-protected.
            value = json.loads(archive.read(name, pwd=b"mind2web"))
            if isinstance(value, list):
                rows.extend(value)
    if not rows:
        raise ValueError("Mind2Web test.zip has no test_task JSON records")
    return rows


def _with_candidate_scores(rows: list[dict], candidate_results: dict) -> None:
    scores, ranks = candidate_results["scores"], candidate_results["ranks"]
    for sample in rows:
        for step in sample["actions"]:
            identity = f"{sample['annotation_id']}_{step['action_uid']}"
            for candidates in (step["pos_candidates"], step["neg_candidates"]):
                for candidate in candidates:
                    key = candidate["backend_node_id"]
                    candidate["score"] = scores[identity][key]
                    candidate["rank"] = ranks[identity][key]


def _train_episode(sample: dict, *, scope: dict[str, str]) -> MemoryEpisode:
    events = [MemoryEvent("action", str(action), role="assistant",
                event_id=f"mind2web-train-{sample['annotation_id']}-{index}",
                extensions={"awm": {"action_repr": str(action)}})
              for index, action in enumerate(sample["action_reprs"])]
    return MemoryEpisode(
        episode_id=f"mind2web-train:{sample['annotation_id']}",
        task=sample["confirmed_task"], events=events, scope=scope,
        extensions={"awm": {"website": sample["website"], "domain": sample["domain"],
            "subdomain": sample["subdomain"], "canonical": True, "end_of_training": True}},
    )


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.env_file and args.env_file.exists():
        _load_env(args.env_file)
    if not os.getenv("AMEM_API_KEY"):
        raise ValueError("Set AMEM_API_KEY in .env.local or the environment")

    env = _env_helpers()
    train_rows = _read_json(args.train_json)
    test_rows = _test_rows(args.test_zip)
    with args.scores.open("rb") as handle:
        candidate_results = pickle.load(handle)
    _with_candidate_scores(train_rows, candidate_results)
    _with_candidate_scores(test_rows, candidate_results)
    train_sample, test_sample = choose_short_pair(train_rows, test_rows)

    scope = {"user_id": "mind2web-smoke"}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_name = os.getenv("AMEM_LLM_MODEL", "qwen-flash")
    config = MemAtlasConfig.from_dict({
        "method": {"method_id": "awm", "params": {
            "profile": "mind2web", "induction_mode": "offline"}},
        "scope": scope,
        "runtime": {"artifact_dir": str(output / "runtime")},
        "providers": {"llm": {
            "model": model_name,
            "base_url": os.getenv("AMEM_BASE_URL"),
            "api_key_env": "AMEM_API_KEY",
            "options": {"extra_body": {"enable_thinking": False}},
        }},
    })
    async with Memory(config) as memory:
        await submit_episode(memory, _train_episode(train_sample, scope=scope))
        contribution = await memory.search(test_sample["confirmed_task"], scope=scope,
            extensions={"awm": {"website": test_sample["website"]}})
        workflow = contribution_text(contribution)
        if not workflow:
            raise RuntimeError("Training trajectory did not produce a workflow file")
        step = test_sample["actions"][0]
        observation, _ = env.get_top_k_obs(step, args.top_k, use_raw=False)
        target = env.get_target_obs_and_act(step)[1]
        messages = [
            {"role": "system", "content": "You are a large language model trained to navigate the web. "
                "Output exactly one next action and wait for the next observation. Use the literal square-bracket "
                "format shown here and output no explanation: `CLICK [id]`, `TYPE [id] [value]`, or "
                "`SELECT [id] [value]`. Here is the action space:\n"
                "1. `CLICK [id]`: Click on an HTML element with its id.\n"
                "2. `TYPE [id] [value]`: Type a string into the element with its id.\n"
                "3. `SELECT [id] [value]`: Select a value for an HTML element with its id."},
            {"role": "user", "content": workflow},
            {"role": "user", "content": f"Task: {test_sample['confirmed_task']}\nTrajectory:\n"
                f"Observation: `{observation}`"},
        ]
        # The public save API writes a checkpoint path; this smoke only needs a
        # read-only method snapshot around the external test-model call.
        before = (await memory._method.save())["workflow_text"]
        response = await memory.services.llm.complete(messages,
            model=model_name, temperature=0.0)
        scores = score_action(str(response), target, step["pos_candidates"])
        after = (await memory._method.save())["workflow_text"]
        result = {
            "benchmark": "Mind2Web",
            "protocol": "one canonical train trajectory; first test action only",
            "quality_claim": False,
            "website": test_sample["website"],
            "train_case": {"annotation_id": train_sample["annotation_id"],
                "action_count": len(train_sample["actions"])},
            "test_case": {"annotation_id": test_sample["annotation_id"],
                "action_count": len(test_sample["actions"])},
            "workflow_before_test": before,
            "workflow_after_test": after,
            "retrieved_memory": contribution.to_dict(),
            "test_observation": observation,
            "model_output": str(response),
            "target_action_for_scoring_only": target,
            "first_step_metrics": scores,
            "full_task_success": None,
            "usage": memory.stats().get("usage", {}),
            "sources": {
                "upstream_revision": "8c0ff8cd11d648c8fceb99e4e42f37e3b75381b1",
                "train_sha256": file_sha256(args.train_json),
                "test_sha256": file_sha256(args.test_zip),
                "scores_sha256": file_sha256(args.scores),
            },
        }
        (output / "mind2web-smoke.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
        (output / "workflow.txt").write_text(workflow, encoding="utf-8")
        return result


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    data = root / ".mematlas" / "awm-validation" / "mind2web-data"
    result = argparse.ArgumentParser(description="One-action Mind2Web smoke through public AWM Memory")
    result.add_argument("--train-json", type=Path, default=data / "train_10.json")
    result.add_argument("--test-zip", type=Path, default=data / "test.zip")
    result.add_argument("--scores", type=Path, default=data / "scores_all_data.pkl")
    result.add_argument("--output", type=Path, default=data / "run")
    result.add_argument("--env-file", type=Path, default=root / ".env.local")
    result.add_argument("--top-k", type=int, default=5)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    report = asyncio.run(run(args))
    print(json.dumps({key: report[key] for key in (
        "benchmark", "protocol", "website", "train_case", "test_case", "first_step_metrics", "usage")},
        ensure_ascii=False, indent=2))
