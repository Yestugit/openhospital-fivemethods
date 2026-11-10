from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

from mematlas.core.errors import ConfigurationError
from mematlas.core.scope import normalize_scope


@dataclass(frozen=True)
class MethodConfig:
    method_id: str
    params: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MethodConfig":
        if set(data) - {"method_id", "params"}:
            raise ConfigurationError("method accepts only method_id and params")
        method_id = data.get("method_id")
        if not method_id:
            raise ConfigurationError("method.method_id is required")
        return cls(str(method_id), data.get("params", {}))


class _MethodOptions(Protocol):
    """Method-owned user settings normalize once at the public config boundary."""

    def _to_method_config(self) -> MethodConfig: ...


def _method_config(value) -> MethodConfig:
    if isinstance(value, MethodConfig):
        return value
    if isinstance(value, str):
        return MethodConfig.from_dict({"method_id": value})
    if isinstance(value, Mapping):
        return MethodConfig.from_dict(value)
    convert = getattr(value, "_to_method_config", None)
    if callable(convert):
        result = convert()
        if isinstance(result, MethodConfig):
            return result
    raise ConfigurationError("method requires MethodConfig, method settings, a mapping or a method ID")


@dataclass(frozen=True)
class RuntimeConfig:
    executor: str = "in_process"
    artifact_dir: str = ".mematlas/artifacts"
    source_roots: Mapping[str, str] = field(default_factory=dict)
    operation_timeout_s: float | None = None
    python_executable: str | None = None
    resource_paths: tuple[str, ...] = ()
    service_url: str | None = None
    service_headers: Mapping[str, str] = field(default_factory=dict)
    service_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        import math
        for name, value in (("operation_timeout_s", self.operation_timeout_s), ("service_timeout_s", self.service_timeout_s)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or value <= 0):
                raise ConfigurationError(f"runtime.{name} must be a positive finite number")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "RuntimeConfig":
        values = dict(data or {})
        values["resource_paths"] = tuple(values.get("resource_paths", ()))
        try:
            return cls(**values)
        except TypeError as exc:
            raise ConfigurationError(f"Invalid runtime configuration: {exc}") from exc


@dataclass(frozen=True)
class MemAtlasConfig:
    method: MethodConfig | _MethodOptions
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    providers: Mapping[str, Any] = field(default_factory=dict)
    scope: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        from mematlas.core.provider_config import resolve_providers
        object.__setattr__(self, "method", _method_config(self.method))
        resolve_providers(self.method.params, self.providers)
        object.__setattr__(self, "scope", normalize_scope(self.scope))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemAtlasConfig":
        unknown = set(data) - {"method", "runtime", "metadata", "providers", "scope"}
        if unknown:
            raise ConfigurationError(f"Unknown Memory configuration fields: {sorted(unknown)}")
        return cls(
            method=_method_config(data.get("method")),
            runtime=RuntimeConfig.from_dict(data.get("runtime")),
            metadata=data.get("metadata", {}),
            providers=data.get("providers", {}),
            scope=data.get("scope", {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> "MemAtlasConfig":
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        elif suffix == ".toml":
            try:
                import tomllib
            except ImportError:  # pragma: no cover - exercised on Python 3.10
                import tomli as tomllib
            data = tomllib.loads(path.read_text(encoding="utf-8-sig"))
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise ConfigurationError("Install mematlas[yaml] to load YAML") from exc
            data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        else:
            raise ConfigurationError(f"Unsupported config format: {suffix}")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": {"method_id": self.method.method_id, "params": dict(self.method.params)},
            "runtime": asdict(self.runtime),
            "scope": dict(self.scope),
            "metadata": dict(self.metadata),
            "providers": {name: value.to_dict() if hasattr(value, "to_dict") else dict(value)
                          for name, value in self.providers.items()},
        }
