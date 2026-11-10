class MemAtlasError(Exception):
    """Base exception for framework failures."""


class ConfigurationError(MemAtlasError):
    pass


class PluginNotFoundError(MemAtlasError):
    pass


class DependencyUnavailableError(MemAtlasError):
    pass


class CapabilityMismatchError(MemAtlasError):
    pass


class MemoryStateError(MemAtlasError):
    pass


class MethodOperationError(MemAtlasError):
    def __init__(self, method_id: str, operation: str, message: str, *, operation_id=None, backend_status=None):
        self.method_id = method_id
        self.operation = operation
        self.operation_id = operation_id
        self.backend_status = backend_status
        super().__init__(f"{method_id}.{operation}: {message}")


class ExecutorError(MemAtlasError):
    pass
