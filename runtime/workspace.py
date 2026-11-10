"""Every Memory starts with isolated storage; load() explicitly restores data."""
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from mematlas.core.errors import ConfigurationError
from mematlas.core.files import atomic_write_json
from mematlas.core.provenance import redact_sensitive


def allocate_workspace(config, plugin):
    run_id = uuid4().hex
    directory = Path(config.runtime.artifact_dir).expanduser().resolve() / "runs" / run_id
    params = dict(config.method.params)
    storage = plugin.descriptor.storage
    paths, namespaces = storage.get("path_parameters", {}), storage.get("namespace_parameters", ())
    if storage.get("kind") == "external_db" and not namespaces:
        raise ConfigurationError("External databases must declare storage.namespace_parameters")
    for name, kind in paths.items():
        if params.get(name):
            target = Path(params[name]).expanduser().resolve()
            is_file = kind == "file" or kind == "auto" and bool(target.suffix)
            root = target.parent if is_file else target
            params[name] = str(root / "runs" / run_id / target.name if is_file else root / "runs" / run_id)
    for name in namespaces:
        params[name] = f"{params.get(name, config.method.method_id)}__{run_id}"
    directory.mkdir(parents=True, exist_ok=False)
    atomic_write_json(directory / "memory-run.json", {
        "interface_version": 3, "run_id": run_id, "requested_config": redact_sensitive(config.to_dict()),
    })
    return replace(config, method=replace(config.method, params=params),
                   runtime=replace(config.runtime, artifact_dir=str(directory))), run_id
