from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping
from dataclasses import dataclass, asdict

from mematlas.core.config import MemAtlasConfig
from mematlas.core.plugin import BaseMemoryMethod, MethodPlugin, MethodServices


@dataclass(frozen=True)
class ExecutionGuarantees:
    cancellation: str = "cooperative"
    backend_may_continue: bool = True
    atomic_rollback: bool = False

    def to_dict(self):
        return asdict(self)


class MethodExecutor(ABC):
    """Creates a Memory method in a selected dependency boundary."""

    name: str
    guarantees = ExecutionGuarantees()

    @abstractmethod
    def create(
        self,
        config: MemAtlasConfig,
        plugin: MethodPlugin,
        services: MethodServices,
    ) -> BaseMemoryMethod:
        raise NotImplementedError

    def check(self, config: MemAtlasConfig) -> Mapping[str, object]:
        return {"executor": self.name, "ready": True}


class InProcessExecutor(MethodExecutor):
    name = "in_process"

    def create(
        self,
        config: MemAtlasConfig,
        plugin: MethodPlugin,
        services: MethodServices,
    ) -> BaseMemoryMethod:
        return plugin.create_method(config.method, services)


class SubprocessExecutor(MethodExecutor):
    name = "subprocess"
    guarantees = ExecutionGuarantees(cancellation="terminate_worker", backend_may_continue=False)

    def check(self, config: MemAtlasConfig) -> Mapping[str, object]:
        import shutil
        import sys
        executable = config.runtime.python_executable or sys.executable
        resolved = shutil.which(executable)
        return {"executor": self.name, "ready": resolved is not None,
                "python_executable": resolved or executable,
                "reason": None if resolved else f"Python executable is unavailable: {executable}"}

    def create(
        self,
        config: MemAtlasConfig,
        plugin: MethodPlugin,
        services: MethodServices,
    ) -> BaseMemoryMethod:
        from mematlas.runtime.execution.subprocess import SubprocessMemoryMethod
        from mematlas.core.serialization import plugin_spec
        services.serializable_spec()  # Fail before launching a process if dependencies cannot be reconstructed.
        return SubprocessMemoryMethod(config, services, plugin_spec(plugin), plugin.descriptor.to_dict())


class ServiceExecutor(MethodExecutor):
    name = "service"
    guarantees = ExecutionGuarantees(cancellation="stop_client_waiting")

    def check(self, config: MemAtlasConfig) -> Mapping[str, object]:
        ready = bool(config.runtime.service_url)
        return {
            "executor": self.name,
            "ready": ready,
            "service_url": config.runtime.service_url,
            "reason": None if ready else "runtime.service_url is required",
        }

    def create(
        self,
        config: MemAtlasConfig,
        plugin: MethodPlugin,
        services: MethodServices,
    ) -> BaseMemoryMethod:
        status = self.check(config)
        if not status["ready"]:
            raise ValueError(str(status["reason"]))
        from mematlas.runtime.execution.service import ServiceMemoryMethod
        from mematlas.core.serialization import plugin_spec
        services.serializable_spec()
        return ServiceMemoryMethod(config, services, plugin_spec(plugin))


class ExecutorRegistry:
    def __init__(self, executors: Mapping[str, MethodExecutor] | None = None) -> None:
        self._executors = dict(executors or {})

    def register(self, executor: MethodExecutor, *, replace: bool = False) -> None:
        if executor.name in self._executors and not replace:
            raise ValueError(f"Executor already registered: {executor.name}")
        self._executors[executor.name] = executor

    def get(self, name: str) -> MethodExecutor:
        try:
            return self._executors[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._executors))
            raise ValueError(f"Unsupported executor {name!r}; available: {available}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._executors))


def default_executors() -> ExecutorRegistry:
    return ExecutorRegistry({
        "in_process": InProcessExecutor(),
        "subprocess": SubprocessExecutor(),
        "service": ServiceExecutor(),
    })
