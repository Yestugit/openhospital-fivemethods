"""Optional LLM bridge for copied AEL reflection/summary helpers."""
from __future__ import annotations

from typing import Any


def llm_call(model: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
    try:
        import litellm
    except ImportError as exc:  # pragma: no cover - optional path
        raise RuntimeError("AEL LLM reflection requires the optional 'litellm' package") from exc
    if model and "/" not in model and model.startswith(("gpt-", "o1", "o3", "o4")):
        model = f"openai/{model}"
    return litellm.completion(model=model, messages=messages, **kwargs)


__all__ = ["llm_call"]
