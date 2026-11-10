"""AEL native memory components loaded under the MemAtlas package namespace."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .native.evolution.meta_controller import MetaController
from .native.memory.store import MemoryStore
from .native.evolution.credit import uniform_credit
from .native.types import MemoryHit, TrajectoryRecord


class AELSettings:
    def __init__(self, profile: str, values: Mapping[str, Any]):
        self.profile, self.values = profile, values


def resolve_settings(params: Mapping[str, Any], variants: Mapping[str, Mapping[str, Any]]) -> AELSettings:
    profile = str(params.get("profile", "default"))
    if profile not in variants:
        raise ValueError(f"Unknown AEL profile {profile!r}; choose from {sorted(variants)}")
    values = deepcopy(dict(variants.get("default", {})))
    values.update(deepcopy(dict(variants[profile])))
    values.update(deepcopy(dict(params)))
    values["profile"] = profile
    return AELSettings(profile, values)


def scope_key(scope: Mapping[str, str]) -> str:
    value = json.dumps(dict(scope), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def task_features(metadata: Mapping[str, Any]):
    supplied = metadata.get("task_features")
    if isinstance(supplied, Mapping):
        values = supplied
    else:
        values = metadata
    from .native.evolution.meta_controller import TaskFeatures
    return TaskFeatures(
        sector_id=int(values.get("sector_id", 0) or 0),
        volatility=float(values.get("volatility", 0.0) or 0.0),
        market_cap_log=float(values.get("market_cap_log", 0.0) or 0.0),
        data_richness=float(values.get("data_richness", 0.0) or 0.0),
        momentum_30d=float(values.get("momentum_30d", 0.0) or 0.0),
        has_options=bool(values.get("has_options", True)),
        has_analyst=bool(values.get("has_analyst", True)),
    )


def create_memory_store(settings: AELSettings) -> MemoryStore:
    if settings.values.get("backend", "local") != "local":
        raise ValueError("AEL backend='xmem' is not supported by the restorable v2 adapter")
    return MemoryStore(
        retrieval_top_k=int(settings.values.get("retrieval_top_k", 5)),
        write_quality_threshold=float(settings.values.get("write_quality_threshold", 0.3)),
        eviction_min_score=float(settings.values.get("eviction_min_score", 0.2)),
        max_entries_per_tier=int(settings.values.get("max_entries_per_tier", 500)),
    )


def create_meta_controller(settings: AELSettings) -> MetaController:
    values = settings.values
    return MetaController(
        # Match the upstream planner pool by default. The memory adapter does
        # not execute planners, but the controller must retain the same arms.
        planner_names=list(values.get("planner_names") or ("sequential", "decompose", "adaptive")),
        tool_config_names=list(values.get("tool_config_names") or ("all", "core", "fundamental", "technical", "sentiment")),
        memory_policy_names=list(values.get("memory_policy_names") or ("none", "recent_window", "full_detailed", "compressed", "aggressive_learner")),
        feature_dim=7,
        linucb_alpha=float(values.get("linucb_alpha", 1.0)),
        evolve_planner=bool(values.get("evolve_planner", True)),
        evolve_tools=bool(values.get("evolve_tools", True)),
        evolve_memory=bool(values.get("evolve_memory", True)),
        per_tool_selection=bool(values.get("per_tool_selection", False)),
        all_tool_names=list(values.get("tool_names", ())),
    )
