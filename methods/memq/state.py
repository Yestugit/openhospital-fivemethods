"""Complete JSON snapshots around the unchanged native NumPy MemoryStore.

MemQ's service checkpoint omits live Q/trace/epoch state and filters its query
caches.  Keep the native store's save/load format, with an adapter sidecar for
the fields that the native checkpoint does not retain.  No provider is called
while restoring a checkpoint, including an intentionally empty query cache.
"""

from __future__ import annotations

import base64
from collections import OrderedDict
from datetime import date, datetime
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping


_SERVICE_FIELDS = (
    "_eligibility_traces", "_accumulated_dq", "_accumulated_dq_count",
    "_q_max", "_current_epoch", "_total_epochs", "dict_memory",
    "query_embeddings", "weight_sim", "weight_q", "normalize_q",
    "dedup_by_task_id", "enable_value_driven", "adjustment_mode",
    "max_keywords", "add_similarity_threshold",
)
_STORE_FILES = ("vectors.npy", "payloads.json")
_STORE_FIELDS = ("_norms", "_payloads", "_id_to_idx", "_dim")


def encode_state(value: Any) -> Any:
    """Retain container types, insertion order and NumPy cache values in JSON."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {
            "kind": "ordered_mapping" if isinstance(value, OrderedDict) else "mapping",
            "items": [[encode_state(key), encode_state(item)] for key, item in value.items()],
        }
    if isinstance(value, (list, tuple)):
        return {"kind": "tuple" if isinstance(value, tuple) else "list",
                "items": [encode_state(item) for item in value]}
    if isinstance(value, (set, frozenset)):
        items = [encode_state(item) for item in value]
        items.sort(key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True))
        return {"kind": "frozenset" if isinstance(value, frozenset) else "set", "items": items}
    if type(value).__module__.split(".", 1)[0] == "numpy":
        if hasattr(value, "ndim") and value.ndim:
            return {"kind": "ndarray", "dtype": str(value.dtype), "shape": list(value.shape),
                    "items": encode_state(value.tolist())}
        return encode_state(value.item())
    if isinstance(value, (datetime, date)):
        return {"kind": "datetime" if isinstance(value, datetime) else "date", "value": value.isoformat()}
    raise TypeError(f"Unsupported MemQ snapshot value: {type(value).__name__}")


def decode_state(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    kind = value["kind"]
    if kind in {"mapping", "ordered_mapping"}:
        items = [(decode_state(key), decode_state(item)) for key, item in value["items"]]
        return OrderedDict(items) if kind == "ordered_mapping" else dict(items)
    if kind == "ndarray":
        import numpy as np

        return np.asarray(decode_state(value["items"]), dtype=value["dtype"]).reshape(value["shape"])
    if kind in {"date", "datetime"}:
        return (datetime if kind == "datetime" else date).fromisoformat(value["value"])
    items = [decode_state(item) for item in value["items"]]
    if kind == "list":
        return items
    if kind == "tuple":
        return tuple(items)
    if kind == "set":
        return set(items)
    if kind == "frozenset":
        return frozenset(items)
    raise ValueError(f"Unknown MemQ snapshot kind: {kind!r}")


def _capture_fields(owner: Any, names: tuple[str, ...]) -> dict[str, Any]:
    return {name: encode_state(getattr(owner, name)) for name in names if hasattr(owner, name)}


def _restore_fields(owner: Any, names: tuple[str, ...], values: Mapping[str, Any]) -> None:
    for name in names:
        if name in values:
            setattr(owner, name, decode_state(values[name]))
        elif hasattr(owner, name):
            delattr(owner, name)


def capture_service(service: Any) -> dict[str, Any]:
    """Export the native store without changing its records, caches or traces."""
    with tempfile.TemporaryDirectory(prefix="mematlas-memq-save-") as directory:
        service._store.save(directory)
        files = {name: base64.b64encode((Path(directory) / name).read_bytes()).decode("ascii")
                 for name in _STORE_FILES}
    result = {
        "schema_version": 1,
        "store": files,
        "store_fields": _capture_fields(service._store, _STORE_FIELDS),
        "fields": _capture_fields(service, _SERVICE_FIELDS),
        "rl_config": encode_state(vars(service.rl_config)) if service.rl_config is not None else None,
    }
    return json.loads(json.dumps(result, ensure_ascii=False, allow_nan=False))


def restore_service(service: Any, state: Mapping[str, Any]) -> None:
    """Load the native store, then restore omitted fields without rebuilding caches."""
    if state.get("schema_version") != 1 or set(state.get("store", {})) != set(_STORE_FILES):
        raise ValueError("Invalid native MemQ snapshot")
    with tempfile.TemporaryDirectory(prefix="mematlas-memq-load-") as directory:
        for name in _STORE_FILES:
            (Path(directory) / name).write_bytes(base64.b64decode(state["store"][name], validate=True))
        store = type(service._store).load(directory)
    _restore_fields(store, _STORE_FIELDS, state["store_fields"])
    service._store = store
    _restore_fields(service, _SERVICE_FIELDS, state["fields"])
    config = state["rl_config"]
    if config is None:
        service.rl_config = None
    else:
        if service.rl_config is None:
            from memq.configs.config import RLConfig

            service.rl_config = RLConfig()
        service.rl_config = type(service.rl_config)(**decode_state(config))
    # AveFactKeyer retains this construction-time configuration separately.
    service.avefact_keyer.max_keywords = service.max_keywords
