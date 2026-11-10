from __future__ import annotations

import asyncio
import inspect
import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Protocol

from mematlas.core.capabilities import MethodDescriptor
from mematlas.core.config import MethodConfig, RuntimeConfig
from mematlas.core.models import (
    Content,
    MemoryContribution,
    MemoryEpisode,
    MemoryEvent,
    MemoryUnit,
    Scope,
)
from mematlas.core.telemetry import UsageLedger


class LLMService(Protocol):
    async def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> str: ...


class EmbeddingService(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class MethodServices:
    llm: LLMService | None = None
    embedder: EmbeddingService | None = None
    objects: dict[str, Any] = field(default_factory=dict)
    source_roots: dict[str, str] = field(default_factory=dict)
    artifact_dir: Path = field(default_factory=lambda: Path(".mematlas/artifacts"))
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("mematlas"))
    usage: UsageLedger = field(default_factory=UsageLedger)
    resource_paths: set[Path] = field(default_factory=set)
    factory_specs: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    provider_configs: dict[str, Any] = field(default_factory=dict)

    def register_resource(self, path: str | Path) -> Path:
        resolved = Path(path).expanduser().resolve()
        self.resource_paths.add(resolved)
        return resolved

    def object(self, key: str, default: Any = None) -> Any:
        return self.objects.get(key, default)

    def source_root(self, method_id: str, fallback: str | None = None) -> Path | None:
        value = self.source_roots.get(method_id) or fallback
        return Path(value).expanduser().resolve() if value else None

    def add_import_root(self, path: str | Path | None) -> None:
        if path is None:
            return
        resolved = str(Path(path).expanduser().resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)

    def serializable_spec(self) -> dict[str, Any]:
        from mematlas.core.serialization import checked_spec, service_spec
        missing = self.objects.keys() - self.factory_specs.keys()
        if missing:
            from mematlas.core.errors import ConfigurationError
            raise ConfigurationError(f"Cannot transfer injected objects {sorted(missing)}; provide factory_specs for isolated execution")
        return {
            "source_roots": dict(self.source_roots),
            "artifact_dir": str(self.artifact_dir),
            "usage_pricing": self.usage.pricing_spec(),
            "provider_configs": {name: value.to_dict() if hasattr(value, "to_dict") else dict(value)
                                 for name, value in self.provider_configs.items()},
            "llm": checked_spec(self.factory_specs["llm"]) if "llm" in self.factory_specs else service_spec(self.llm),
            "embedder": checked_spec(self.factory_specs["embedder"]) if "embedder" in self.factory_specs else service_spec(self.embedder),
            "objects": {name: checked_spec(spec) for name, spec in self.factory_specs.items() if name not in {"llm", "embedder"}},
        }

    @classmethod
    def from_spec(cls, data: Mapping[str, Any]) -> "MethodServices":
        services = cls(
            source_roots=dict(data.get("source_roots", {})),
            artifact_dir=Path(data.get("artifact_dir", ".mematlas/artifacts")),
            usage=UsageLedger(**dict(data.get("usage_pricing", {}))),
            provider_configs=dict(data.get("provider_configs", {})),
        )
        from mematlas.core.serialization import construct
        for path in services.source_roots.values():
            services.add_import_root(path)
        for name in ("llm", "embedder"):
            if data.get(name):
                provider = construct(data[name])
                if hasattr(provider, "usage"):
                    provider.usage = services.usage
                setattr(services, name, provider)
        services.objects = {name: construct(spec) for name, spec in data.get("objects", {}).items()}
        return services


@dataclass(frozen=True)
class MethodContext:
    run_id: str
    method_config: MethodConfig
    runtime_config: RuntimeConfig
    metadata: Mapping[str, Any] = field(default_factory=dict)


async def call_maybe_async(
    function: Callable[..., Any], *args: Any, run_sync_in_thread: bool = False, **kwargs: Any
) -> Any:
    if inspect.iscoroutinefunction(function):
        return await function(*args, **kwargs)
    if run_sync_in_thread:
        result = await asyncio.to_thread(function, *args, **kwargs)
    else:
        result = function(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


class BaseMemoryMethod(ABC):
    """Base contract implemented by every Memory method."""

    async def setup(self, context: MethodContext) -> None:
        return None

    async def _observe(self, event: MemoryEvent) -> None:
        """Method-internal online ingestion, shared with native algorithms."""
        return None

    async def index(self, *, scope: Scope) -> list[dict[str, Any]]:
        # A default view describes the access actually exposed by this adapter.
        # It makes no claim about the method's internal storage hierarchy.
        return [{"level_ref": "default", "description": "Default searchable memory view",
                 "scope": dict(scope), "granularity": "method-native memory entries"}]

    async def add(self, unit: MemoryUnit, *, learning_episode: MemoryEpisode | None = None) -> None:
        if learning_episode is not None:
            await self._learn(learning_episode, observed_event_ids=frozenset())
            await self._after_learning(learning_episode)
        else:
            for event in unit.events:
                await self._observe(event)

    async def search(self, query: Content, *, level_ref: str | None, top_k: int | None, scope: Scope,
                     extensions: Mapping[str, Any]) -> MemoryContribution:
        raise NotImplementedError

    async def _learn(self, episode: MemoryEpisode, *, observed_event_ids: frozenset[str]) -> None:
        """Native learning implementation, reused by add and task completion."""
        raise NotImplementedError

    async def _after_learning(self, episode: MemoryEpisode) -> None:
        return None

    async def episode_end(self, episode_ref: str, *, scope: Scope, status: str,
                          learning_episode: MemoryEpisode | None = None) -> None:
        """Receive one task boundary; native learning waits for real feedback."""
        if learning_episode is not None:
            await self._learn(learning_episode, observed_event_ids=frozenset())
            await self._after_learning(learning_episode)

    async def detail(self, memory_id: str, *, scope: Scope) -> MemoryContribution:
        raise NotImplementedError

    async def update(self, memory_id: str, content: Content, *, scope: Scope) -> None:
        raise NotImplementedError

    async def delete(self, memory_id: str, *, scope: Scope) -> None:
        raise NotImplementedError

    async def flush(self) -> None:
        return None

    async def _freeze(self) -> None:
        """Pause background mutations at a safe boundary, preserving pending work.

        Internal executor hook; must not trigger extraction or drain queued learning.
        load() while frozen must leave the backend frozen until _thaw().
        """
        return None

    async def _thaw(self) -> None:
        """Resume scheduling after frozen inspection or restoration."""
        return None

    async def reset(self, scope: Scope) -> None:
        raise NotImplementedError

    async def save(self) -> Mapping[str, Any]:
        raise NotImplementedError

    async def load(self, state: Mapping[str, Any]) -> None:
        raise NotImplementedError

    async def command(self, name: str, payload: Mapping[str, Any]) -> Any:
        raise NotImplementedError(f"Method does not expose command {name!r}")

    async def close(self) -> None:
        return None


class MethodPlugin(ABC):
    descriptor: MethodDescriptor
    # Evaluation-only adapter; the Memory runtime never imports evaluation.
    qa_factory: str | None = None

    @abstractmethod
    def create_method(
        self, config: MethodConfig, services: MethodServices
    ) -> BaseMemoryMethod:
        raise NotImplementedError(f"{type(self).__name__} must implement create_method()")

    def validate_config(self, config: MethodConfig) -> list[str]:
        from mematlas.core.validation import validate_parameters
        return validate_parameters(config.params, self.descriptor.config_schema)

    def validate_input(self, operation: str, extensions: Mapping[str, Any]) -> list[str]:
        """Methods declare only the extra inputs their actual operations need."""
        from mematlas.core.validation import validate_parameters
        return validate_parameters(extensions, self.descriptor.input_schemas.get(operation, {}),
                                   path=f"{self.descriptor.method_id}.{operation}.extensions")

    def healthcheck(self, config: MethodConfig, services: MethodServices) -> Mapping[str, Any]:
        from importlib import metadata
        import os
        source_root = services.source_root(self.descriptor.method_id)
        config_errors = self.validate_config(config)
        source_required = bool(self.descriptor.source_subdir)
        source_available = bool(source_root and source_root.is_dir())
        missing_packages = []
        for package in self.descriptor.requirements.get("python", []):
            try:
                metadata.version(package)
            except metadata.PackageNotFoundError:
                missing_packages.append(package)
        missing_services = []
        from mematlas.core.provider_config import resolve_providers
        providers = resolve_providers(config.params, services.provider_configs)
        if config.params.get("implementation_mode", "standard") != "offline":
            for name in self.descriptor.requirements.get("services", []):
                if name == "llm" and config.params.get("infer") is False:
                    continue
                supplied = services.embedder if name == "embedding" else services.llm
                provider = providers["embedding" if name == "embedding" else "llm"]
                if supplied is None and not (provider.model and os.environ.get(provider.api_key_env)):
                    missing_services.append(name)
        ready = not (config_errors or missing_packages or missing_services) and (source_available or not source_required)
        return {
            "status": "ready" if ready else "unavailable",
            "ready": ready,
            "method_id": self.descriptor.method_id,
            "source_root": str(source_root) if source_root else None,
            "source_required": source_required,
            "source_available": source_available,
            "recommended_executor": self.descriptor.recommended_executor,
            "config_errors": config_errors,
            "missing_packages": missing_packages,
            "missing_services": missing_services,
        }
