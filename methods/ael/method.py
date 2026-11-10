from __future__ import annotations

from copy import deepcopy
import json
import logging
import math
from pathlib import Path
from typing import Any, Mapping

from mematlas.core.config import MethodConfig
from mematlas.core.models import MemoryContribution, MemoryEpisode, MemoryEvent, content_text
from mematlas.core.files import atomic_write_json
from mematlas.core.plugin import BaseMemoryMethod, MethodContext, MethodPlugin, MethodServices, call_maybe_async
from mematlas.core.scope import normalize_scope
from mematlas.methods.manifest import load_method_descriptor

from .core import AELSettings, create_memory_store, create_meta_controller, resolve_settings, scope_key, task_features
from .native.evolution.credit import portfolio_credit, structural_credit, uniform_credit
from .native.types import PlanStep, ToolCall, TrajectoryRecord
from .state import decode_state, detached_json, encode_state, capture_components, restore_components

DESCRIPTOR = load_method_descriptor(__file__)
LOGGER = logging.getLogger("mematlas.methods.ael.adapter")


class AELMemory(BaseMemoryMethod):
    def __init__(self, config: MethodConfig, services: MethodServices):
        self.params = dict(config.params)
        self.services = services
        self.settings: AELSettings = resolve_settings(self.params, DESCRIPTOR.config_schema["x-variants"])
        storage_root = Path(str(self.params.get("storage_path") or (self.services.artifact_dir / "ael"))).expanduser().resolve()
        self._storage_path = storage_root / "adapter_state.json"
        self._stores: dict[str, Any] = {}
        self._controllers: dict[str, Any] = {}
        self._scopes: dict[str, dict[str, str]] = {}
        self._outcomes: dict[str, list[float]] = {}
        self._last_retrieval: dict[str, dict[str, Any]] = {}
        self._pending_day_episodes: dict[str, int] = {}
        self._trajectory_history: dict[str, list[dict[str, Any]]] = {}
        self._day_indices: dict[str, int] = {}

    async def setup(self, context: MethodContext) -> None:
        if self.params.get("backend", "local") != "local":
            raise ValueError("AEL backend='xmem' cannot be restored by Memory v2")
        if self._storage_path.is_file():
            await self.load(json.loads(self._storage_path.read_text(encoding="utf-8")))

    async def _persist_native(self) -> None:
        atomic_write_json(self._storage_path, await self.save())

    async def _components(self, scope: Mapping[str, str]):
        owner = normalize_scope(scope)
        key = scope_key(owner)
        if key not in self._stores:
            self._stores[key] = create_memory_store(self.settings)
            self._controllers[key] = create_meta_controller(self.settings)
            self._scopes[key] = owner
            self._outcomes[key] = []
            self._trajectory_history[key] = []
            self._day_indices[key] = 0
        return key, self._stores[key], self._controllers[key]

    @staticmethod
    def _ael_extensions(values: Mapping[str, Any] | None) -> dict[str, Any]:
        data = dict(values or {})
        nested = data.get("ael")
        return dict(nested) if isinstance(nested, Mapping) else data

    @staticmethod
    def _query(query: Any, extensions: Mapping[str, Any]) -> dict[str, Any]:
        values = AELMemory._ael_extensions(extensions)
        result = dict(values.get("memory_query", {})) if isinstance(values.get("memory_query"), Mapping) else {}
        if not result:
            result = {key: values[key] for key in ("task_type", "ticker", "sector", "planner", "market_regime", "tool_name") if values.get(key) not in (None, "")}
        if not result:
            result["task_type"] = values.get("task_type", "portfolio_allocation")
        return result

    async def index(self, *, scope):
        return [{"level_ref": ref, "scope": dict(scope), "description": description, "granularity": granularity}
                for ref, description, granularity in (
                    ("default", "Policy-selected memory tiers", "native combined context"),
                    ("episodic", "Task execution experience", "experience entries"),
                    ("semantic", "Consolidated knowledge", "knowledge entries"),
                    ("procedural", "Reusable procedures", "procedure entries"))]

    async def search(self, query, *, level_ref=None, top_k, scope, extensions) -> MemoryContribution:
        key, store, controller = await self._components(scope)
        values = self._ael_extensions(extensions)
        features = task_features(values)
        warmup = bool(values.get("_warmup", False))
        frozen = bool(values.get("frozen", False) or extensions.get("read_only", False))
        selection = ({"planner": "sequential", "tool_config": "__all__", "memory_policy": "none"}
                     if warmup else controller.select(features, ticker=str(values.get("ticker")) if values.get("ticker") else None, frozen=frozen))
        policy = controller.get_memory_policy(selection["memory_policy"])
        if level_ref not in {None, "default"}:
            policy = {**policy, "enabled_tiers": [level_ref]}
        query_struct = self._query(query, extensions)
        hits = [] if warmup else await call_maybe_async(store.retrieve_with_strategy, query_struct, policy, run_sync_in_thread=True)
        limit = top_k if top_k is not None else int(policy.get("retrieval_top_k", self.settings.values.get("retrieval_top_k", 5)))
        if limit >= 0:
            hits = list(hits)[:limit]
        decision_id = f"{key}:{len(self._last_retrieval.get(key, {}).get('ids', [])) + 1}"
        rows = []
        ids = []
        for hit in hits:
            memory_id = str(hit.memory_id)
            ids.append(memory_id)
            rows.append({
                "content": str(hit.content), "memory_id": memory_id,
                "score": float(hit.relevance_score), "source_event_ids": [],
                "memory_type": str(hit.tier), "level_ref": str(hit.tier),
                "metadata": {"method_id": "ael", "tier": str(hit.tier),
                             "ael_controller_selection": deepcopy(selection),
                             "ael_task_features": deepcopy(vars(features)),
                             "ael_decision_id": decision_id},
            })
        decision = {"method_id": "ael", "decision_id": decision_id,
                    "controller_selection": deepcopy(selection), "task_features": deepcopy(vars(features))}
        self._last_retrieval[key] = {"ids": ids, "decision": decision, "selection": deepcopy(selection), "features": vars(features)}
        if not extensions.get("read_only"):
            await self._persist_native()
        return MemoryContribution(items=rows, extensions={"ael": {
            "memory_decisions": [decision], "memory_policy": selection["memory_policy"],
            "controller_selection": deepcopy(selection), "task_features": deepcopy(vars(features)),
            "native_hit_count": len(hits), "_warmup": warmup,
        }})

    async def _observe(self, event: MemoryEvent) -> None:
        self._scopes[scope_key(event.scope)] = dict(event.scope)

    @staticmethod
    def _episode_data(episode: MemoryEpisode) -> dict[str, Any]:
        values = dict(episode.extensions)
        nested = values.get("ael")
        if isinstance(nested, Mapping):
            values.update(nested)
        return values

    @staticmethod
    def _score(episode: MemoryEpisode, values: Mapping[str, Any]) -> float:
        score = values.get("outcome_score", episode.reward)
        if score is None:
            score = values.get("reward")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValueError("AEL requires a finite numeric reward or ael.outcome_score")
        return float(score)

    @staticmethod
    def _trajectory(episode: MemoryEpisode, values: Mapping[str, Any]) -> str:
        text = "\n".join(f"{event.role or event.kind.value}: {content_text(event.content)}" for event in episode.events or () if content_text(event.content).strip())
        if not text:
            text = str(values.get("task") or episode.task or "")
        if not text.strip():
            raise ValueError("AEL requires episode trajectory text or task")
        return text

    @staticmethod
    def _applied_contexts(episode: MemoryEpisode, values: Mapping[str, Any]) -> tuple[list[tuple[dict[str, Any], Any]], list[str]]:
        contexts, ids, seen = [], [], set()
        for contribution in episode.applied_memory:
            ext = contribution.extensions.get("ael", {}) if isinstance(contribution.extensions, Mapping) else {}
            decisions = ext.get("memory_decisions", []) if isinstance(ext, Mapping) else []
            if not decisions and isinstance(ext.get("controller_selection"), Mapping):
                decisions = [{"controller_selection": ext["controller_selection"], "task_features": ext.get("task_features", {})}]
            for decision in decisions:
                if not isinstance(decision, Mapping):
                    continue
                selected = decision.get("controller_selection")
                if not isinstance(selected, Mapping):
                    continue
                ident = str(decision.get("decision_id") or repr(sorted(selected.items())))
                if ident not in seen:
                    seen.add(ident)
                    contexts.append((dict(selected), task_features(decision.get("task_features", {}))))
            for item in contribution.items:
                if not isinstance(item, Mapping):
                    continue
                memory_id = item.get("memory_id")
                if memory_id and str(memory_id) not in ids:
                    ids.append(str(memory_id))
                metadata = item.get("metadata", {})
                if isinstance(metadata, Mapping):
                    selected = metadata.get("ael_controller_selection")
                    if isinstance(selected, Mapping):
                        ident = str(metadata.get("ael_decision_id") or repr(sorted(selected.items())))
                        if ident not in seen:
                            seen.add(ident)
                            contexts.append((dict(selected), task_features(metadata.get("ael_task_features", {}))))
        selection = values.get("controller_selection")
        if isinstance(selection, Mapping) and not contexts:
            contexts.append((dict(selection), task_features(values)))
        return contexts, ids

    async def _learn(self, episode: MemoryEpisode, *, observed_event_ids=frozenset()) -> None:
        values = self._episode_data(episode)
        if values.get("read_only") or values.get("frozen"):
            return
        score = self._score(episode, values)
        rich = isinstance(values.get("tool_evaluations"), Mapping) and any(values.get(name) is not None for name in ("ticker", "prediction", "actual"))
        warmup = bool(values.get("_warmup", False))
        trajectory = "" if rich or warmup else self._trajectory(episode, values)
        key, store, controller = await self._components(episode.scope)
        contexts, retrieved_ids = self._applied_contexts(episode, values)
        if warmup and not contexts:
            contexts = [({"planner": "sequential", "tool_config": "__all__", "memory_policy": "none"}, task_features(values))]
        scores = self._outcomes.setdefault(key, [])
        if self.settings.values.get("runner_mode", "portfolio") == "episode":
            scores.append(score)
            self._trajectory_history.setdefault(key, []).append({"outcome_score": score, "memory_retrievals": []})
        episode_num = len(scores)
        useful = score > (float(__import__("numpy").mean(scores[-20:])) if len(scores) >= 5 and self.settings.values.get("usefulness_mode") == "differential" else 0.0)
        credits = values.get("module_credits")
        if not isinstance(credits, Mapping):
            credits = self._credit_for_episode(episode, values, score, rich)
        for selection, features in contexts:
            await call_maybe_async(controller.update, dict(selection), features, float(__import__("numpy").clip((score + 1.0) / 2.0, .05, .95)), dict(credits), planner_alpha=values.get("planner_alpha"), alpha_z=values.get("alpha_z"), run_sync_in_thread=True)
        for memory_id in retrieved_ids:
            await call_maybe_async(store.mark_useful, memory_id, useful, run_sync_in_thread=True)
        await call_maybe_async(store.increment_episode, run_sync_in_thread=True)
        interval = int(self.settings.values.get("distill_interval", 10))
        if episode_num > 0 and interval > 0 and episode_num % interval == 0:
            await call_maybe_async(store.auto_distill, min_episodes=int(self.settings.values.get("distill_min_episodes", 8)), run_sync_in_thread=True)
            await call_maybe_async(store.promote_to_procedural, min_samples=int(self.settings.values.get("promotion_min_samples", 8)), min_confidence=float(self.settings.values.get("promotion_min_confidence", .6)), run_sync_in_thread=True)
        if warmup:
            await self._persist_native()
            return
        tool_evaluations = values.get("tool_evaluations")
        if rich:
            await call_maybe_async(store.add_episode_experience, ticker=str(values.get("ticker", "unknown")), date=str(values.get("date", "?")), planner_id=str(values.get("planner_id", values.get("planner", ""))), tool_evaluations={str(name): dict(value) for name, value in tool_evaluations.items() if isinstance(value, Mapping)}, prediction=dict(values.get("prediction") or {}), actual=dict(values.get("actual") or {}), outcome_score=score, sector=str(values.get("sector", "")), run_sync_in_thread=True)
        else:
            feedback = episode.feedback
            content = trajectory + (f"\nOutcome: {feedback}" if feedback else "")
            await call_maybe_async(store.add_episodic, content, {"ticker": values.get("ticker", ""), "sector": values.get("sector", ""), "planner": values.get("planner", ""), "task_type": values.get("task_type", ""), "outcome_score": score, "market_regime": values.get("market_regime", "")}, quality=max(.3, min(1.0, (score + 1.0) / 2.0)), run_sync_in_thread=True)
        await self._persist_native()

    @staticmethod
    def _credit_for_episode(episode: MemoryEpisode, values: Mapping[str, Any], score: float, rich: bool) -> dict[str, float]:
        """Select the available native credit path without requiring AEL's Agent loop.

        Portfolio episodes carry the same realized per-ticker/tool data used by
        AEL's portfolio credit function. For generic episodes, structural credit
        is the deterministic fallback; uniform credit remains an explicit option.
        """
        mode = str(values.get("credit_method", "fcc"))
        if mode == "uniform":
            return uniform_credit()
        if rich and mode in {"fcc", "portfolio"}:
            prediction = values.get("prediction") or {}
            actual = values.get("actual") or {}
            weights = prediction.get("weights", {}) if isinstance(prediction, Mapping) else {}
            per_ticker = values.get("per_ticker", {})
            if not isinstance(per_ticker, Mapping):
                per_ticker = {}
            returns = {str(t): float(v.get("return", 0.0)) for t, v in per_ticker.items() if isinstance(v, Mapping)}
            tool_outputs = values.get("tool_outputs", {})
            if not isinstance(tool_outputs, Mapping):
                tool_outputs = {}
            if weights and returns:
                return portfolio_credit(weights=dict(weights), ticker_returns=returns,
                                        tool_outputs={str(k): dict(v) for k, v in tool_outputs.items() if isinstance(v, Mapping)},
                                        memory_hits=[], equal_weight=1.0 / max(len(returns), 1),
                                        planner_alpha=values.get("planner_alpha"))
        if mode == "structural":
            trace = TrajectoryRecord(outcome_score=score)
            trace.plan_steps = [PlanStep(i, "event", "", success=True) for i, _ in enumerate(episode.events or ())]
            return structural_credit(trace)
        return uniform_credit()

    async def save(self) -> Mapping[str, Any]:
        import numpy as np
        return detached_json({"schema_version": 2, "method_id": "ael", "profile": self.settings.profile,
            "scopes": {key: {"scope": self._scopes[key], "native": capture_components(self._stores[key], self._controllers[key])} for key in self._stores},
            "observed_scopes": self._scopes,
            "adapter": encode_state({"outcomes": self._outcomes, "last_retrieval": self._last_retrieval, "pending_day_episodes": self._pending_day_episodes, "trajectory_history": self._trajectory_history, "day_indices": self._day_indices}),
            "numpy_random_state": encode_state(np.random.get_state())})

    async def load(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != 2 or state.get("method_id") != "ael":
            raise ValueError("Invalid AEL state")
        stores, controllers, scopes = {}, {}, {}
        for key, row in state.get("scopes", {}).items():
            scope = normalize_scope(row["scope"])
            if scope_key(scope) != key:
                raise ValueError("AEL scope identity mismatch")
            store, controller = create_memory_store(self.settings), create_meta_controller(self.settings)
            restore_components(store, controller, row["native"])
            stores[key], controllers[key], scopes[key] = store, controller, scope
        adapter = decode_state(state["adapter"])
        self._stores, self._controllers, self._scopes = stores, controllers, scopes
        self._outcomes = adapter["outcomes"]
        self._last_retrieval = adapter["last_retrieval"]
        self._pending_day_episodes = adapter.get("pending_day_episodes", {})
        self._trajectory_history = adapter["trajectory_history"]
        self._day_indices = adapter["day_indices"]
        import numpy as np
        np.random.set_state(decode_state(state["numpy_random_state"]))

    async def reset(self, scope) -> None:
        key = scope_key(scope)
        self._stores.pop(key, None); self._controllers.pop(key, None); self._scopes.pop(key, None)
        self._outcomes.pop(key, None); self._last_retrieval.pop(key, None); self._pending_day_episodes.pop(key, None)
        self._trajectory_history.pop(key, None); self._day_indices.pop(key, None)
        await self._persist_native()

    async def _after_learning(self, episode: MemoryEpisode) -> None:
        options = episode.extensions.get("ael", {})
        await self.command("complete_learning_episode", {
            "scope": episode.scope, "_warmup": bool(options.get("_warmup", False))})

    async def command(self, name: str, payload: Mapping[str, Any]) -> Any:
        if name == "bandit_state":
            scope = payload.get("scope")
            key = scope_key(scope) if isinstance(scope, Mapping) else str(payload.get("scope_key", ""))
            if key not in self._controllers:
                return {}
            controller = self._controllers[key]
            return {**deepcopy(controller.state_dict()), "memory_policies": deepcopy(controller.memory_registry.policies)}
        if name == "complete_learning_episode":
            if payload.get("_warmup"): return {"skipped": True}
            scope = normalize_scope(payload["scope"]); key = scope_key(scope)
            pending = self._pending_day_episodes.get(key, 0) + 1; self._pending_day_episodes[key] = pending
            if pending >= int(self.settings.values.get("bars_per_day", 4)):
                result = await self._nightly_memory({"scope": scope}); self._pending_day_episodes[key] = 0
                await self._persist_native()
                return result
            await self._persist_native()
            return {"pending_day_episodes": pending}
        if name == "finish_learning_days":
            results = []
            for key, count in list(self._pending_day_episodes.items()):
                if count: results.append(await self._nightly_memory({"scope_key": key})); self._pending_day_episodes[key] = 0
            await self._persist_native()
            return results
        if name in {"end_day", "nightly_memory"}:
            result = await self._nightly_memory(payload)
            await self._persist_native()
            return result
        raise ValueError(f"Unknown AEL command {name!r}")

    async def _nightly_memory(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if payload.get("frozen") or payload.get("read_only") or payload.get("_warmup"): return {"skipped": True}
        scope = payload.get("scope"); key = scope_key(scope) if isinstance(scope, Mapping) else str(payload.get("scope_key", ""))
        if not key and len(self._stores) == 1: key = next(iter(self._stores))
        if key not in self._stores: raise ValueError("AEL nightly_memory requires an initialized scope")
        store, controller = self._stores[key], self._controllers[key]
        day = self._day_indices.get(key, 0) + 1; self._day_indices[key] = day
        interval = 1 if int(self.settings.values.get("bars_per_day", 1)) >= 4 else 2
        consolidated = day % interval == 0
        if consolidated:
            await call_maybe_async(store.auto_distill, min_episodes=int(self.settings.values.get("distill_min_episodes", 8)), run_sync_in_thread=True)
            await call_maybe_async(store.promote_to_procedural, min_samples=int(self.settings.values.get("promotion_min_samples", 8)), min_confidence=float(self.settings.values.get("promotion_min_confidence", .6)), run_sync_in_thread=True)
        controller.clear_morning_overrides(); self._pending_day_episodes[key] = 0
        return {"scope_key": key, "day_index": day, "consolidated": consolidated, "policy_added": None}

    async def close(self) -> None:
        self._stores.clear(); self._controllers.clear()


class AELPlugin(MethodPlugin):
    descriptor = DESCRIPTOR
    qa_factory = "mematlas.methods.ael.qa:AELQA"

    def validate_config(self, config: MethodConfig) -> list[str]:
        errors = super().validate_config(config)
        if config.params.get("backend", "local") == "xmem": errors.append("AEL backend='xmem' is unsupported")
        return errors

    def create_method(self, config: MethodConfig, services: MethodServices) -> BaseMemoryMethod:
        return AELMemory(config, services)


PLUGIN = AELPlugin()
