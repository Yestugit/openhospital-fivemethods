"""Small native MemCon-compatible store used by the v2 adapter.

The upstream project is an agent framework.  MemAtlas keeps the G-Memory
record protocol and a dependency-free policy-compatible fallback behind a
narrow object protocol.  The fallback exposes observable action selection,
discounted episode credit and generalized plans; it does not replace the
upstream Chroma/LLM implementation.  A native-compatible object can be
injected through ``memcon.instance`` or ``memcon.service_factory``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from mematlas.methods.core_utils import lexical_score


@dataclass
class MemConSettings:
    profile: str = "default"
    values: Mapping[str, Any] = field(default_factory=dict)


def resolve_settings(params: Mapping[str, Any], variants: Mapping[str, Mapping[str, Any]]) -> MemConSettings:
    profile = str(params.get("profile", "default"))
    if profile not in variants:
        raise ValueError(f"Unknown MemCon profile {profile!r}; choose from {sorted(variants)}")
    values = dict(variants.get("default", {}))
    values.update(dict(variants[profile]))
    values.update(dict(params))
    values["profile"] = profile
    return MemConSettings(profile, values)


def scope_key(scope: Mapping[str, str]) -> str:
    payload = json.dumps(dict(scope), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _goal_type(task: str) -> str:
    lowered = task.lower()
    for verb in ("puttwo", "clean", "heat", "cool", "examine", "look at", "put"):
        if verb in lowered:
            return verb
    return "other"


_ACTION_SPECS: tuple[tuple[str, int, int], ...] = (
    ("retrieve", 1, 3),
    ("retrieve", 2, 5),
    ("retrieve", 3, 8),
    ("plan_inject", 1, 3),
    ("re_retrieve", 2, 5),
    ("consolidate", 0, 5),
    ("forget", 0, 0),
    ("noop", 0, 0),
)


class _CompatMDP:
    """Small observable MDP history compatible with the adapter contract.

    The full upstream MDP tracks a richer environment state.  Keeping the
    selected action history here lets the adapter preserve action provenance
    and gives the fallback policy a real learning signal.
    """

    def __init__(self) -> None:
        self.state: dict[str, Any] = {}
        self._episode_actions: list[tuple[str, str]] = []

    def reset(self, *, goal_type: str, task_index: int) -> None:
        self.state = {"goal_type": goal_type, "task_index": task_index, "step": 0}
        self._episode_actions = []

    def observe(self, **values: Any) -> dict[str, Any]:
        self.state.update(values)
        return self.state

    def record_action(self, action: str) -> None:
        key = json.dumps(self.state, sort_keys=True, separators=(",", ":"))
        self._episode_actions.append((key, action))
        self.state["step"] = int(self.state.get("step", 0)) + 1

    def get_episode_actions(self) -> list[tuple[str, str]]:
        return list(self._episode_actions)

    def clear_episode(self) -> None:
        self._episode_actions = []


class MemConNative:
    """Dependency-free local implementation of the MemCon memory protocol."""

    def __init__(self, *, settings: MemConSettings, storage_path: Path):
        self.settings = settings
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []
        self.insights: list[str] = []
        self.policy: dict[str, dict[str, float]] = {}
        self.task: str | None = None
        self.task_description: str | None = None
        self.actions: list[dict[str, Any]] = []
        self._step_history: list[dict[str, str]] = []
        self._insights_cache: list[str] = []
        self._success_plans: dict[str, list[str]] = {}
        self._task_index = 0
        self._goal_type = "other"
        self._mdp = _CompatMDP()
        self._load()

    def _load(self) -> None:
        path = self.storage_path / "native.json"
        if not path.is_file():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.records = list(payload.get("records", []))
        self.insights = list(payload.get("insights", []))
        self.policy = {str(k): dict(v) for k, v in payload.get("policy", {}).items()}
        self._success_plans = {str(k): list(v) for k, v in payload.get("success_plans", {}).items()}
        self._task_index = int(payload.get("task_index", 0))

    def _persist(self) -> None:
        path = self.storage_path / "native.json"
        path.write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")

    def init_task_context(self, task_main: str, task_description: str | None = None, **_: Any) -> None:
        self.task = str(task_main)
        self.task_description = task_description
        self.actions = []
        self._step_history = []
        self._insights_cache = []
        self._task_index += 1
        self._goal_type = _goal_type(self.task)
        self._mdp.reset(goal_type=self._goal_type, task_index=self._task_index)

    def move_memory_state(self, action: str, observation: str, **kwargs: Any) -> None:
        self.actions.append({"action": str(action), "observation": str(observation),
                             "reward": float(kwargs.get("reward", 0.0) or 0.0),
                             "done": bool(kwargs.get("done", False))})
        self._step_history.append({"action": str(action), "observation": str(observation)})
        self._mdp.observe(step=len(self._step_history), is_stuck=(
            len(self._step_history) > 1 and self._step_history[-2]["action"] == str(action)))

    def _policy_state(self, goal: str) -> dict[str, Any]:
        state = self.policy.setdefault(goal, {"retrieves": 0.0, "reward": 0.0})
        # Migrate the old counter-only state without invalidating snapshots.
        state.setdefault("q_values", {})
        state.setdefault("visits", {})
        return state

    def _select_action(self, goal: str, *, has_records: bool) -> tuple[str, int, int]:
        state = self._policy_state(goal)
        q_values = state["q_values"]
        visits = state["visits"]
        total = sum(int(value) for value in visits.values())
        candidates = _ACTION_SPECS[:-1] if has_records else _ACTION_SPECS[:3]
        # Warm-start each useful action once, then use an upper-confidence score.
        for name, _, _ in candidates:
            if int(visits.get(name, 0)) == 0:
                return name, next(k for k, spec in enumerate(_ACTION_SPECS) if spec[0] == name), 0
        scored = []
        for name, top_k, insight_k in candidates:
            count = max(1, int(visits.get(name, 0)))
            score = float(q_values.get(name, 0.0)) + 1.4 * math.sqrt(math.log(total + 2) / count)
            scored.append((score, name, top_k, insight_k))
        _, name, top_k, insight_k = max(scored, key=lambda row: (row[0], row[1]))
        return name, next(k for k, spec in enumerate(_ACTION_SPECS) if spec[0] == name), insight_k

    @staticmethod
    def _generalize_actions(actions: list[str]) -> list[str]:
        return [re.sub(r"(shelf|drawer|cabinet|countertop|desk|bed|sink|fridge|microwave|stove)\s*\d+", r"[\1]", action, flags=re.IGNORECASE) for action in actions[:15]]

    def retrieve_memory(self, query_task: str, *, successful_topk: int = 1,
                        failed_topk: int = 0, insight_topk: int = 5,
                        threshold: float = 0.3, **_: Any) -> tuple[list[dict], list[dict], list[str]]:
        goal = _goal_type(str(query_task))
        scored = []
        for record in self.records:
            score = lexical_score(str(query_task), str(record.get("task", "")))
            if score >= threshold:
                scored.append((score, record))
        scored.sort(key=lambda pair: (-pair[0], str(pair[1].get("memory_id", ""))))
        action_name, action_index, action_insight_k = self._select_action(goal, has_records=bool(self.records))
        self._mdp.observe(goal_type=goal, mem_size=len(self.records), plan_available=goal in self._success_plans)
        self._mdp.record_action(action_name)
        state = self._policy_state(goal)
        state["retrieves"] = float(state.get("retrieves", 0.0)) + 1.0
        if action_name == "noop":
            success, failure, insights = [], [], []
        elif action_name == "forget":
            self.insights.clear()
            success, failure, insights = [], [], []
        else:
            limit_success = action_insight_k if action_name != "plan_inject" else max(1, action_insight_k)
            success = [{**record, "score": score} for score, record in scored if record.get("success") is True][:min(successful_topk, max(1, limit_success))]
            failure = [{**record, "score": score} for score, record in scored if record.get("success") is False][:failed_topk]
            insights = list(self.insights[:min(insight_topk, max(1, action_insight_k))])
            if action_name == "re_retrieve":
                retry = f"{query_task} alternative approach"
                extra = sorted(((lexical_score(retry, str(record.get("task", ""))), record) for record in self.records), key=lambda pair: -pair[0])
                success = [{**record, "score": score} for score, record in extra if record.get("success") is True][:successful_topk]
            if action_name in {"plan_inject", "retrieve"} and goal in self._success_plans:
                insights.insert(0, "[Proven plan]\n" + "\n".join(f"{i + 1}. {step}" for i, step in enumerate(self._success_plans[goal])))
        insights = insights[:insight_topk]
        self._insights_cache = insights
        self._persist()
        return success, failure, insights

    def save_task_context(self, *, label: bool, feedback: str | None = None, **_: Any) -> dict[str, Any]:
        if not self.task:
            raise ValueError("MemCon task context is not initialized")
        trajectory = "\n".join(f"> {row['action']}\n{row['observation']}" for row in self._step_history)
        memory_id = hashlib.sha256(f"{self.task}\n{trajectory}\n{len(self.records)}".encode()).hexdigest()[:16]
        record = {"memory_id": memory_id, "task": self.task, "task_description": self.task_description,
                  "trajectory": trajectory, "success": bool(label), "feedback": feedback,
                  "key_steps": self._key_steps(trajectory),
                  "source_event_ids": [], "memory_type": "success_experience" if label else "failure_experience"}
        self.records.append(record)
        if label and self._step_history:
            self.insights.append("; ".join(row["action"] for row in self._step_history[:8]))
            self._success_plans[self._goal_type] = self._generalize_actions([row["action"] for row in self._step_history])
        goal = self._goal_type
        state = self._policy_state(goal)
        reward = (1.0 if label else -0.5) + (max(0.0, 1.0 - len(self._step_history) / 30.0) * 0.3 if label else 0.0)
        state["reward"] = float(state.get("reward", 0.0)) + reward
        actions = self._mdp.get_episode_actions()
        for distance, (_, action) in enumerate(reversed(actions)):
            visits = state["visits"]
            q_values = state["q_values"]
            visits[action] = int(visits.get(action, 0)) + 1
            target = reward * (0.9 ** distance)
            q_values[action] = float(q_values.get(action, 0.0)) + 0.15 * (target - float(q_values.get(action, 0.0)))
        self._mdp.clear_episode()
        self._persist()
        return record

    @staticmethod
    def _key_steps(trajectory: str) -> str:
        actions = [line[2:].strip() for line in trajectory.splitlines() if line.startswith("> ")]
        return "\n".join(f"{i + 1}. {action}" for i, action in enumerate(actions[:8]))

    def backward(self, success: bool, **_: Any) -> None:
        # The policy is updated in save_task_context, matching the upstream
        # wrapper.  backward only clears the temporary insight credit cache.
        self._insights_cache = []
        self._persist()

    def snapshot(self) -> dict[str, Any]:
        return {"schema_version": 2, "records": self.records, "insights": self.insights,
                "policy": self.policy, "task": self.task, "task_description": self.task_description,
                "actions": self.actions, "step_history": self._step_history,
                "insights_cache": self._insights_cache, "success_plans": self._success_plans,
                "task_index": self._task_index, "goal_type": self._goal_type,
                "mdp_actions": self._mdp.get_episode_actions(), "mdp_state": self._mdp.state}

    def restore(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") not in {1, 2}:
            raise ValueError("Invalid native MemCon state")
        self.records = list(state.get("records", []))
        self.insights = list(state.get("insights", []))
        self.policy = {str(k): dict(v) for k, v in state.get("policy", {}).items()}
        self.task = state.get("task")
        self.task_description = state.get("task_description")
        self.actions = list(state.get("actions", []))
        self._step_history = list(state.get("step_history", []))
        self._insights_cache = list(state.get("insights_cache", []))
        self._success_plans = {str(k): list(v) for k, v in state.get("success_plans", {}).items()}
        self._task_index = int(state.get("task_index", 0))
        self._goal_type = str(state.get("goal_type", _goal_type(self.task or "")))
        self._mdp.state = dict(state.get("mdp_state", {}))
        self._mdp._episode_actions = [tuple(row) for row in state.get("mdp_actions", [])]
        self._persist()

    def reset(self) -> None:
        self.records.clear()
        self.insights.clear()
        self.policy.clear()
        self._success_plans.clear()
        self.task = self.task_description = None
        self.actions.clear()
        self._step_history.clear()
        self._insights_cache.clear()
        self._mdp.clear_episode()
        self._persist()


def create_native(settings: MemConSettings, storage_path: Path) -> MemConNative:
    return MemConNative(settings=settings, storage_path=storage_path)
