"""Select scene adapters without putting scene-specific work in the runner."""
from importlib import import_module
from collections.abc import Mapping


SCENE_FACTORIES = {
    "openhospital": "mematlas.integrations.openhospital.application:OpenHospitalScene",
}


def _factory(spec):
    if not isinstance(spec, Mapping):
        return None
    declaration = SCENE_FACTORIES.get(spec.get("kind"))
    if declaration is None:
        return None
    if isinstance(declaration, str):
        module, name = declaration.split(":", 1)
        declaration = getattr(import_module(module), name)
    return declaration


def scene_for(config, registry):
    factory = _factory(config.benchmark)
    return factory(config, registry=registry) if factory is not None else None


def resolve_paths(spec, directory):
    if not isinstance(spec, Mapping):
        return spec
    factory = _factory(spec)
    keys = getattr(factory, "path_options", ("path",))
    result = dict(spec)
    from pathlib import Path
    for key in keys:
        if result.get(key) and not Path(result[key]).is_absolute():
            result[key] = str((directory / result[key]).resolve())
    return result
