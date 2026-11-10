"""Explicit construction specs for plugins and services crossing process boundaries."""

import importlib
import inspect
import json
from typing import Any, Mapping

from mematlas.core.errors import ConfigurationError


def resolve_reference(reference: str) -> Any:
    module_name, separator, name = reference.partition(":")
    if not separator or module_name == "__main__" or "<locals>" in name:
        raise ConfigurationError("Factories must use an importable module:qualified_name reference")
    value = importlib.import_module(module_name)
    for part in name.split("."):
        value = getattr(value, part)
    return value


def checked_spec(spec: Mapping[str, Any]) -> dict:
    result = dict(spec)
    if not isinstance(result.get("factory"), str):
        raise ConfigurationError("A construction spec requires factory=module:qualified_name")
    try:
        json.dumps(result, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ConfigurationError("Construction specs must contain JSON values") from exc
    return result


def construct(spec: Mapping[str, Any]) -> Any:
    factory = resolve_reference(str(spec["factory"]))
    return factory(**dict(spec.get("kwargs", {}))) if spec.get("call", True) else factory


def plugin_spec(plugin) -> dict:
    if hasattr(plugin, "construction_spec"):
        return checked_spec(plugin.construction_spec())
    module_name = type(plugin).__module__
    module = importlib.import_module(module_name)
    if getattr(module, "PLUGIN", None) is plugin:
        return checked_spec({"factory": f"{module_name}:PLUGIN", "call": False})
    name = type(plugin).__qualname__
    if module_name == "__main__" or "<locals>" in name:
        raise ConfigurationError("Isolated plugins must be importable; define the plugin in a module or supply construction_spec()")
    try:
        inspect.signature(type(plugin)).bind()
    except TypeError as exc:
        raise ConfigurationError("Plugin requires constructor arguments; supply construction_spec()") from exc
    return checked_spec({"factory": f"{module_name}:{name}"})


def service_spec(service) -> dict | None:
    if service is None:
        return None
    if hasattr(service, "construction_spec"):
        return checked_spec(service.construction_spec())
    raise ConfigurationError(
        f"Cannot transfer {type(service).__name__} to an isolated executor. "
        "Supply an importable construction_spec() or MethodServices.factory_specs."
    )
