from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Iterable, Mapping


def tokens(text: str) -> list[str]:
    return re.findall(r"[\w\u4e00-\u9fff]+", text.lower())


def lexical_score(query: str, document: str) -> float:
    query_counts = Counter(tokens(query))
    document_counts = Counter(tokens(document))
    if not query_counts or not document_counts:
        return 0.0
    overlap = sum((query_counts & document_counts).values())
    denominator = math.sqrt(sum(query_counts.values()) * sum(document_counts.values()))
    return overlap / denominator if denominator else 0.0


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        values = []
        for item in content:
            if isinstance(item, Mapping):
                if item.get("type") == "text":
                    values.append(str(item.get("text", "")))
                elif item.get("type") == "image_url":
                    image = item.get("image_url", {})
                    values.append(f"[image: {image.get('url', '')}]")
            else:
                values.append(str(item))
        return " ".join(value for value in values if value)
    return str(content or "")


def parse_json_object(text: str) -> Mapping[str, Any] | None:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.I | re.S)
    try:
        value = json.loads(candidate)
    except (TypeError, json.JSONDecodeError):
        match = re.search(r"\{.*\}", candidate, flags=re.S)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, Mapping) else None


def join_messages(messages: Iterable[Mapping[str, Any]]) -> str:
    lines = []
    for message in messages:
        text = content_text(message.get("content"))
        if text:
            lines.append(f"{message.get('role', 'user')}: {text}")
    return "\n".join(lines)


def offline_embedding(text: str, dimensions: int = 128) -> list[float]:
    """Deterministic lexical vectors for explicitly configured offline tests."""
    import hashlib
    values = [0.0] * dimensions
    for token in tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        values[int.from_bytes(digest[:4], "big") % dimensions] += 1.0 if digest[4] & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values] if norm else values


def vision_content(text: str, images: Iterable[str]) -> str | list[dict[str, Any]]:
    """Preserve image bytes in model calls while keeping original evidence URIs."""
    import base64
    import mimetypes
    from pathlib import Path

    uris = list(dict.fromkeys(images))
    if not uris:
        return text
    content = [{"type": "text", "text": text}]
    for uri in uris:
        url = uri
        if not uri.startswith(("https://", "http://", "data:")):
            path = Path(uri).expanduser()
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
            url = f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")
        content.extend([
            {"type": "text", "text": f"Image evidence: {uri}"},
            {"type": "image_url", "image_url": {"url": url}},
        ])
    return content
