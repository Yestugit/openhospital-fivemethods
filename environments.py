from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class MethodEnvironmentSpec:
    method_id: str
    source_subdir: str
    import_checks: tuple[str, ...]
    install_mode: str
    source_import_subdirs: tuple[str, ...] = ()
    external_services: tuple[str, ...] = ()
    compatible_scenes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "source_subdir": self.source_subdir,
            "import_checks": list(self.import_checks),
            "install_mode": self.install_mode,
            "source_import_subdirs": list(self.source_import_subdirs),
            "external_services": list(self.external_services),
            "compatible_scenes": list(self.compatible_scenes),
        }


def _builtin_environments():
    from mematlas.core.registry import default_registry
    result = {}
    for plugin in default_registry().plugins():
        descriptor = plugin.descriptor
        if descriptor.method_id.startswith("baseline_"):
            continue
        package = type(plugin).__module__.split(".")[-2]
        properties = descriptor.config_schema.get("properties", {})
        profiles = properties.get("profile", properties.get("benchmark", {})).get("enum", ())
        result[descriptor.method_id] = MethodEnvironmentSpec(
            descriptor.method_id, package, ("mematlas",), "integrated_core",
            external_services=tuple(descriptor.requirements.get("services", ())),
            compatible_scenes=tuple(profiles) or descriptor.route_hints,
        )
    return result


ENVIRONMENT_SPECS = _builtin_environments()


def environment_spec(method_id: str) -> MethodEnvironmentSpec:
    try:
        return ENVIRONMENT_SPECS[method_id]
    except KeyError as exc:
        raise ValueError(f"No environment specification for {method_id!r}") from exc


def inspect_method_environment(
    method_id: str,
    *,
    python_executable: str | Path | None = None,
    methods_root: str | Path | None = None,
    source_root: str | Path | None = None,
) -> Mapping[str, Any]:
    spec = environment_spec(method_id)
    executable = str(python_executable or sys.executable)
    root = Path(methods_root or Path(__file__).resolve().parent / "methods").resolve()
    resolved_source_root = Path(source_root).expanduser().resolve() if source_root else root / spec.source_subdir
    import_roots = [str(resolved_source_root / subdir) for subdir in spec.source_import_subdirs]
    code = (
        "import importlib.util,json,sys;"
        f"names={json.dumps(list(spec.import_checks))};"
        f"roots={json.dumps(import_roots)};"
        "[sys.path.insert(0,root) for root in reversed(roots)];"
        "result={};"
        "\nfor name in names:\n"
        " try:\n"
        "  item=importlib.util.find_spec(name); result[name]={'available':item is not None,'origin':getattr(item,'origin',None)}\n"
        " except Exception as exc:\n"
        "  result[name]={'available':False,'error':type(exc).__name__+': '+str(exc)}\n"
        "print(json.dumps({'python':sys.executable,'version':sys.version,'imports':result}))"
    )
    try:
        completed = subprocess.run(
            [executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=os.environ.copy(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            **spec.to_dict(),
            "ready": False,
            "python_executable": executable,
            "source_root": str(resolved_source_root),
            "source_available": resolved_source_root.is_dir(),
            "error": f"{type(exc).__name__}: {exc}",
        }
    payload: dict[str, Any] = {}
    if completed.returncode == 0:
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            payload = {}
    imports = payload.get("imports", {})
    ready = (
        completed.returncode == 0
        and resolved_source_root.is_dir()
        and all(imports.get(name, {}).get("available") for name in spec.import_checks)
    )
    return {
        **spec.to_dict(),
        "ready": ready,
        "python_executable": payload.get("python", executable),
        "python_version": payload.get("version"),
        "source_root": str(resolved_source_root),
        "source_available": resolved_source_root.is_dir(),
        "imports": imports,
        "probe_returncode": completed.returncode,
        "probe_stderr": completed.stderr.strip() or None,
    }
