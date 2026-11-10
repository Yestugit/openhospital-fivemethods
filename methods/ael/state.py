"""JSON-safe snapshots for AEL's native store and bandits."""
from __future__ import annotations
from collections import OrderedDict
from datetime import date, datetime
import json
from typing import Any, Mapping


def encode_state(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {"kind": "ordered_mapping" if isinstance(value, OrderedDict) else "mapping",
                "items": [[encode_state(k), encode_state(v)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return {"kind": "tuple" if isinstance(value, tuple) else "list", "items": [encode_state(v) for v in value]}
    if isinstance(value, (set, frozenset)):
        items = [encode_state(v) for v in value]
        items.sort(key=lambda v: json.dumps(v, sort_keys=True, ensure_ascii=False))
        return {"kind": "frozenset" if isinstance(value, frozenset) else "set", "items": items}
    if type(value).__module__.split(".", 1)[0] == "numpy":
        if getattr(value, "ndim", 0):
            return {"kind": "ndarray", "dtype": str(value.dtype), "shape": list(value.shape), "items": encode_state(value.tolist())}
        return encode_state(value.item())
    if type(value).__name__ == "TaskFeatures":
        return {"kind": "task_features", "fields": encode_state(vars(value))}
    if isinstance(value, (date, datetime)):
        return {"kind": "datetime" if isinstance(value, datetime) else "date", "value": value.isoformat()}
    raise TypeError(f"Unsupported AEL snapshot value: {type(value).__name__}")


def decode_state(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    kind = value.get("kind")
    if kind in {"mapping", "ordered_mapping"}:
        pairs = [(decode_state(k), decode_state(v)) for k, v in value["items"]]
        return OrderedDict(pairs) if kind == "ordered_mapping" else dict(pairs)
    if kind == "ndarray":
        import numpy as np
        return np.asarray(decode_state(value["items"]), dtype=value["dtype"]).reshape(value["shape"])
    if kind == "task_features":
        from .native.evolution.meta_controller import TaskFeatures
        return TaskFeatures(**decode_state(value["fields"]))
    if kind in {"date", "datetime"}:
        return (datetime if kind == "datetime" else date).fromisoformat(value["value"])
    if kind in {"list", "tuple", "set", "frozenset"}:
        items = [decode_state(v) for v in value["items"]]
        return {"list": list, "tuple": tuple, "set": set, "frozenset": frozenset}[kind](items)
    raise ValueError(f"Unknown AEL snapshot kind: {kind!r}")


def detached_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


_STORE_FIELDS = ("_top_k", "_write_threshold", "_eviction_min", "_max_per_tier", "_read_only")
_CONTROLLER_FIELDS = ("evolve_planner", "evolve_tools", "evolve_memory", "per_tool_selection", "_all_tool_names", "_episodes_elapsed", "_morning_overrides", "_fixed_planner", "_fixed_tools", "_fixed_memory")


def capture_components(store: Any, controller: Any) -> dict[str, Any]:
    return detached_json({
        "schema_version": 1,
        "memory": encode_state(store.state_dict()),
        "memory_fields": encode_state({name: getattr(store, name) for name in _STORE_FIELDS}),
        "controller": encode_state(controller.state_dict()),
        "controller_fields": encode_state({name: getattr(controller, name) for name in _CONTROLLER_FIELDS}),
        "memory_policies": encode_state(controller.memory_registry.policies),
        "planner_fields": encode_state({"alpha_0": controller.planner_bandit.alpha_0, "_total_pulls": controller.planner_bandit._total_pulls}),
    })


def restore_components(store: Any, controller: Any, state: Mapping[str, Any]) -> None:
    if state.get("schema_version") != 1:
        raise ValueError("Invalid native AEL snapshot")
    store._entries.clear()
    store.load_state_dict(decode_state(state["memory"]))
    for name, value in decode_state(state["memory_fields"]).items():
        setattr(store, name, value)
    native = decode_state(state["controller"])
    planner = native["planner_bandit"]
    controller.planner_bandit = type(controller.planner_bandit)(list(planner["arms"]), dim=planner["dim"], alpha=planner["alpha"])
    controller.tool_bandit = type(controller.tool_bandit)(list(native["tool_bandit"]))
    controller.memory_bandit = type(controller.memory_bandit)(list(native["memory_bandit"]))
    controller.load_state_dict(native)
    for name, value in decode_state(state["controller_fields"]).items():
        setattr(controller, name, value)
    controller.memory_registry._policies = decode_state(state["memory_policies"])
    for name, value in decode_state(state["planner_fields"]).items():
        setattr(controller.planner_bandit, name, value)
