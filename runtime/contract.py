"""Runtime enforcement of a method's declared public capabilities."""
from mematlas.core.capabilities import Operation
from mematlas.core.contracts import validate_modalities, validate_scope
from mematlas.core.errors import CapabilityMismatchError
from mematlas.core.scope import normalize_scope


class MemoryContract:
    def __init__(self, config, descriptor):
        self.config, self.descriptor = config, descriptor
        self.capabilities = descriptor.capabilities

    def require(self, operation: Operation):
        if operation not in self.capabilities.operations:
            raise CapabilityMismatchError(f"{self.descriptor.method_id}: unsupported operation {operation.value}")
        if operation in {Operation.SAVE, Operation.LOAD} and not self.capabilities.supports_persistence:
            raise CapabilityMismatchError("Method does not support persistence")

    def scope(self, explicit=None, recorded=None):
        scope = normalize_scope(explicit if explicit is not None else recorded if recorded is not None else self.config.scope)
        if recorded is not None and scope != recorded:
            raise ValueError("Operation scope differs from the recorded data's scope")
        validate_scope(self.capabilities, scope)
        return scope

    def event(self, event):
        if self.capabilities.accepted_events and event.kind not in self.capabilities.accepted_events:
            raise CapabilityMismatchError(f"Unsupported event kind: {event.kind.value}")
        validate_modalities(self.capabilities, event.content)
