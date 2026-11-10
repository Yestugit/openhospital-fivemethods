"""Internal storage identifiers and source-linked MemoryOS context formatting."""
from hashlib import sha256
import json
import re

from mematlas.core.models import MemoryContribution


_WINDOWS_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def safe_storage_user_id(value: str) -> str:
    if (0 < len(value) <= 120 and re.fullmatch(r"[A-Za-z0-9._-]+", value)
            and value.rstrip(". ") == value and value.split(".", 1)[0].upper() not in _WINDOWS_RESERVED_NAMES):
        return value
    return "mematlas-" + sha256(value.encode("utf-8")).hexdigest()[:32]


def format_contribution(short_term, retrieved, profile, user_id, top_k):
    items = []

    def add(text, kind, identity=None, score=None, sources=(), timestamp=None):
        item = {"content": text, "memory_type": kind, "source_event_ids": list(sources),
                "metadata": {"method_id": "memoryos"}}
        if identity:
            item["memory_id"] = str(identity)
        if score is not None:
            item["score"] = score
        if timestamp:
            item["metadata"]["timestamp"] = timestamp
        items.append(item)

    for pair in short_term:
        text = f"User: {pair.get('user_input', '')}\nAssistant: {pair.get('agent_response', '')}\nTime: {pair.get('timestamp', '')}"
        identity = sha256(json.dumps(pair, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:24]
        add(text, "short_term", "short-term-" + identity,
            sources=pair.get("meta_data", {}).get("mematlas_event_ids", []), timestamp=pair.get("timestamp"))
    for page in retrieved.get("retrieved_pages", []):
        text = (f"User: {page.get('user_input', '')}\nAssistant: {page.get('agent_response', '')}\n"
                f"Time: {page.get('timestamp', '')}\nConversation chain overview: {page.get('meta_info', 'N/A')}")
        add(text, "mid_term", page.get("memory_id") or page.get("page_id"), page.get("score"),
            page.get("meta_data", {}).get("mematlas_event_ids", []), page.get("timestamp"))
    for key, kind in (("retrieved_user_knowledge", "user_knowledge"),
                      ("retrieved_assistant_knowledge", "assistant_knowledge")):
        for row in retrieved.get(key, []):
            add(str(row.get("knowledge", "")), kind, row.get("id"), row.get("score"), timestamp=row.get("timestamp"))
    if profile and str(profile).lower() != "none":
        add(str(profile), "user_profile", "profile-" + user_id)

    if top_k is not None:
        items = items[:top_k]
    groups = (
        ("short_term_history", "Short-term conversation", {"short_term"}),
        ("mid_term_retrieval", "Historical memory", {"mid_term"}),
        ("user_background", "User background", {"user_profile", "user_knowledge"}),
        ("assistant_knowledge", "Assistant knowledge", {"assistant_knowledge"}),
    )
    context = []
    for kind, label, types in groups:
        indices = [i for i, item in enumerate(items) if item["memory_type"] in types]
        if indices:
            context.append({"kind": kind, "content": f"[{label}]\n" + "\n".join(items[i]["content"] for i in indices),
                            "source_item_indices": indices})
    return MemoryContribution(items=items, context=context)
