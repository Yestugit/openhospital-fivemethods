from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

from .config import load_settings


_settings = load_settings()
_invoke_capacity = threading.BoundedSemaphore(_settings.invoke_max_concurrency)
_chat_capacity = threading.BoundedSemaphore(_settings.chat_max_concurrency)
_evaluate_capacity = threading.BoundedSemaphore(_settings.evaluate_max_concurrency)


@contextmanager
def bounded_route(path: str) -> Iterator[None]:
    capacity = _capacity_for_path(path)
    if capacity is None:
        yield
        return
    if not capacity.acquire(blocking=False):
        raise PermissionError("服务繁忙，请稍后重试。")
    try:
        yield
    finally:
        capacity.release()


def _capacity_for_path(path: str) -> threading.BoundedSemaphore | None:
    if path == "/invoke":
        return _invoke_capacity
    if path == "/llm/chat":
        return _chat_capacity
    if path in {"/evaluate", "/evaluate/case"}:
        return _evaluate_capacity
    return None

