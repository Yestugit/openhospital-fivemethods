"""ReasoningBank external-feedback, single-trajectory Memory v3 adapter."""
from .config import ReasoningBankConfig, _DESCRIPTOR
from .method import PLUGIN

SUPPORTED_OPERATIONS = tuple(sorted(operation.value for operation in _DESCRIPTOR.capabilities.operations))

__all__ = ["ReasoningBankConfig", "SUPPORTED_OPERATIONS"]
