from __future__ import annotations

from typing import Any

__all__ = ["ExamService"]


def __getattr__(name: str) -> Any:
    if name != "ExamService":
        raise AttributeError(name)
    from .service import ExamService

    return ExamService
