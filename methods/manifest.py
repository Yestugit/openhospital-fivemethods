from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from mematlas.core.capabilities import MethodDescriptor


REQUIRED_MANIFEST_KEYS = frozenset({
    "method_id",
    "name",
    "summary",
    "capabilities",
    "config_schema",
})


def manifest_path(method_file: str | Path) -> Path:
    return Path(method_file).resolve().parent / "method.yaml"


def read_method_manifest(method_file: str | Path) -> Mapping[str, Any]:
    path = manifest_path(method_file)
    text = path.read_text(encoding="utf-8-sig")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                f"{path} uses YAML syntax; install MemAtlas with the 'yaml' extra"
            ) from exc
        data = yaml.safe_load(text)
    if not isinstance(data, Mapping):
        raise ValueError(f"Method manifest must contain an object: {path}")
    missing = sorted(REQUIRED_MANIFEST_KEYS - set(data))
    if missing:
        raise ValueError(f"Method manifest {path} is missing: {', '.join(missing)}")
    return data


def load_method_descriptor(method_file: str | Path) -> MethodDescriptor:
    data = dict(read_method_manifest(method_file))
    data.setdefault("manifest_version", 1)
    descriptor = MethodDescriptor.from_dict(data)
    errors = descriptor.validate()
    if errors:
        raise ValueError("; ".join(errors))
    return descriptor
