"""AEL three-tier financial memory and policy controller.

Provide numeric ``MemoryUnit.feedback['reward']`` (or ``extensions['ael']['outcome_score']``)
from the external portfolio evaluator.  Structured query fields and tool
evaluations live under ``extensions['ael']``. Submit units with ``add`` and
notify task completion with ``episode_end``. Native day counting runs internally;
commands remain available for genuine external day boundaries and inspection.
"""
from .config import AELConfig, _DESCRIPTOR
from .method import PLUGIN

SUPPORTED_OPERATIONS = tuple(sorted(operation.value for operation in _DESCRIPTOR.capabilities.operations))
__all__ = ["AELConfig", "SUPPORTED_OPERATIONS"]
