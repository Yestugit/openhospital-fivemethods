"""Side-effect-free checks shared by Memory and experiments."""
from dataclasses import replace

from mematlas.core.config import MemAtlasConfig
from mematlas.core.contracts import validate_scope
from mematlas.core.plugin import MethodServices
from mematlas.core.provider_config import resolve_providers
from mematlas.core.provenance import redact_sensitive
from mematlas.core.registry import default_registry
from mematlas.runtime.execution import default_executors


def check_config(config, *, registry=None, services=None, requirements=None, inspect_environment=True):
    config = config if isinstance(config, MemAtlasConfig) else MemAtlasConfig.from_dict(config)
    registry = registry or default_registry()
    plugin = registry.get(config.method.method_id)
    providers = resolve_providers(config.method.params, config.providers)
    supplied = services or MethodServices()
    supplied = replace(supplied, provider_configs=providers,
                       source_roots={**supplied.source_roots, **config.runtime.source_roots})
    errors = plugin.validate_config(config.method)
    if plugin.descriptor.interface_version != 3:
        errors.append(f"{plugin.descriptor.method_id}: method requires migration to Memory interface v3")
    try:
        validate_scope(plugin.descriptor.capabilities, config.scope)
    except Exception as exc:
        errors.append(str(exc))
    if requirements is not None:
        errors.extend(requirements.check(plugin.descriptor))
    executor = default_executors().get(config.runtime.executor)
    executor_status = dict(executor.check(config))
    health = {"ready": True, "status": "not_inspected", "inspection": "disabled"}
    if config.runtime.executor in {"subprocess", "service"} and not errors:
        try:
            supplied.serializable_spec()
        except Exception as exc:
            errors.append(str(exc))
    if inspect_environment:
        if config.runtime.executor == "subprocess":
            if not errors and executor_status.get("ready"):
                from mematlas.runtime.execution.subprocess import probe_environment
                health = probe_environment(config, plugin, supplied)
            else:
                health = {"ready": False, "status": "unavailable", "inspection": "worker",
                          "reason": "Resolve configuration/executor errors before environment inspection"}
        else:
            health = dict(plugin.healthcheck(config.method, supplied))
    return {"valid": not errors, "ready": not errors and bool(health.get("ready")) and bool(executor_status.get("ready")),
            "errors": errors, "health": health, "executor": executor_status,
            "execution_guarantees": executor.guarantees.to_dict(),
            "effective_config": redact_sensitive({**config.to_dict(),
                "providers": {name: value.to_dict() for name, value in providers.items()}})}
