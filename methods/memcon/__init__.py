"""MemCon: event-driven task memory with policy feedback.

Submit action/observation units through ``add(MemoryUnit(...))``; include the
task and real boolean ``feedback['success']`` when available. Call ``search``
when the Agent needs context and ``episode_end`` once the task finishes. Search returns
success_experience, failure_experience and insight items plus a ``memcon``
decision extension.  The package fallback is offline and deterministic;
``memcon.instance`` can inject the original native backend.
"""
from .config import MemConConfig, _DESCRIPTOR
from .method import PLUGIN

SUPPORTED_OPERATIONS = tuple(sorted(operation.value for operation in _DESCRIPTOR.capabilities.operations))

__all__ = ["MemConConfig", "SUPPORTED_OPERATIONS"]
