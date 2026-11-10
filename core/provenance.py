from __future__ import annotations

import hashlib
import json
from importlib import metadata
from pathlib import Path
import subprocess
from typing import Any, Mapping


_SECRET_KEY_SUFFIXES = (
    "apikey",
    "accesstoken",
    "authtoken",
    "bearertoken",
    "clientsecret",
    "password",
    "secret",
    "token",
)


def redact_sensitive(value: Any, *, key: str = "") -> Any:
    """Return a report-safe copy of nested configuration data."""

    compact_key = "".join(character for character in key.lower() if character.isalnum())
    sensitive = False if compact_key.endswith("env") else any(
        compact_key.endswith(suffix) for suffix in _SECRET_KEY_SUFFIXES
    )
    if sensitive and value is not None:
        return "<redacted>"
    if isinstance(value, Mapping):
        return {
            str(child_key): redact_sensitive(child_value, key=str(child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_sensitive(item) for item in value]
    return value


def git_revision(path: str | Path) -> str | None:
    root = Path(path).expanduser().resolve()
    if not root.exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def package_version(package: str = "mematlas", *, source_fallback: str = "0.1.0+source") -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return source_fallback


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256()
    for chunk in encoder.iterencode(value):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()
