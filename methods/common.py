from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Mapping, Sequence

from mematlas.core.models import Scope
from mematlas.core.scope import normalize_scope, scope_key


def scope_identity(scope: Scope) -> dict[str, str]:
    """Keep every supplied ownership field, including session isolation."""
    return normalize_scope(scope) or {"run_id": scope_key({})}


def first_value(data: Any, names: Sequence[str], default: Any = None) -> Any:
    if isinstance(data, Mapping):
        for name in names:
            if name in data and data[name] is not None:
                return data[name]
    for name in names:
        value = getattr(data, name, None)
        if value is not None:
            return value
    return default


def unwrap_results(data: Any) -> list[Any]:
    if data is None:
        return []
    if isinstance(data, Mapping):
        for key in ("results", "memories", "items", "data"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        return [data]
    if isinstance(data, (list, tuple)):
        return list(data)
    return [data]


def normalize_items(
    data: Any,
    *,
    method_id: str,
    memory_type: str | None = None,
    content_fields: Sequence[str] = ("memory", "content", "text", "fact", "summary"),
) -> list[dict[str, Any]]:
    items = []
    for raw in unwrap_results(data):
        if hasattr(raw, "model_dump"):
            raw_data = raw.model_dump(mode="json")
        elif isinstance(raw, Mapping):
            raw_data = dict(raw)
        elif hasattr(raw, "__dict__"):
            raw_data = dict(vars(raw))
        else:
            raw_data = {"raw_type": type(raw).__name__}
        content = first_value(raw_data, content_fields, default="")
        if isinstance(content, (dict, list)):
            content = json.dumps(content, ensure_ascii=False, default=str)
        if not str(content).strip():
            continue
        score = first_value(raw_data, ("score", "similarity", "relevance"))
        try:
            score = float(score) if score is not None else None
        except (TypeError, ValueError):
            score = None
        created_at = first_value(raw_data, ("created_at", "timestamp", "created"))
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                created_at = None
        if not isinstance(created_at, datetime):
            created_at = None
        nested_metadata = raw_data.get("metadata")
        provenance_sources = [raw_data]
        if isinstance(nested_metadata, Mapping):
            provenance_sources.append(nested_metadata)
        source_event_ids: list[str] = []
        for source in provenance_sources:
            for key in ("source_event_ids", "mematlas_event_ids", "event_ids"):
                values = source.get(key)
                if isinstance(values, str):
                    values = [values]
                if isinstance(values, Sequence):
                    for value in values:
                        text = str(value)
                        if text and text not in source_event_ids:
                            source_event_ids.append(text)
        # Preserve method data without repeating content or returning vectors.
        used = set(content_fields) | {"id", "uuid", "memory_id", "key", "score", "similarity", "relevance",
            "created_at", "timestamp", "created", "embedding", "source_event_ids", "mematlas_event_ids", "event_ids"}
        item = {"content": str(content), "source_event_ids": source_event_ids,
                "metadata": {"method_id": method_id, **{key: value for key, value in raw_data.items() if key not in used}}}
        identity = first_value(raw_data, ("id", "uuid", "memory_id", "key"))
        if identity is not None:
            item["memory_id"] = str(identity)
        if score is not None:
            item["score"] = score
        if created_at is not None:
            item["created_at"] = created_at.isoformat()
        if memory_type is not None:
            item["memory_type"] = memory_type
        items.append(item)
    return items


def event_images(content) -> list[str]:
    return [block["uri"] for block in content if block["type"] == "image"] if isinstance(content, list) else []
