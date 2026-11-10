"""Public contract for the external environment used by AWM web QA."""

from __future__ import annotations

from typing import Awaitable, Protocol, TypedDict


class BrowserState(TypedDict, total=False):
    """Actual state returned by an external environment reset or step.

    ``observation`` and ``done`` are required on every state. A terminal state
    must also include the environment's actual boolean ``success`` result.
    """

    observation: str
    done: bool
    success: bool


class AWMEnvironment(Protocol):
    """Synchronous or asynchronous browser adapter consumed by ``AWMQA``.

    ``reset`` starts an unfinished task, ``step`` executes one raw action, and
    ``close`` releases the browser or simulator resources.
    """

    def reset(self) -> BrowserState | Awaitable[BrowserState]: ...

    def step(self, action: str) -> BrowserState | Awaitable[BrowserState]: ...

    def close(self) -> None | Awaitable[None]: ...


__all__ = ["AWMEnvironment", "BrowserState"]
