"""Read-only source checks for adapters that import a separate upstream tree."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from mematlas.core.plugin import MethodServices


def inspect_source(
    services: MethodServices,
    *,
    method_id: str,
    fallback: str | Path,
    required_files: Sequence[str],
    expected_revision: str | None = None,
    required_text: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Inspect source availability without importing providers or starting a DB.

    A Git checkout with a known revision must match ``expected_revision``.
    Vendored trees may instead provide UPSTREAM.json with a ``revision`` and
    a ``files`` mapping of relative paths to SHA-256 hashes. Every listed file
    and every required file must be covered by that inventory. Other archives
    are accepted with an explicitly unverified revision. A Git marker is only
    inspected at the source root, so an archive inside MemAtlas does not
    inherit MemAtlas's unrelated revision.

    ``required_text`` is for an adapter's declared compatibility patch; this
    function reports missing markers and never edits or applies that patch.
    """
    root = services.source_root(method_id, str(fallback))
    assert root is not None
    missing_files = [name for name in required_files if not (root / name).is_file()]
    source_available = root.is_dir() and not missing_files
    errors = []
    if not root.is_dir():
        errors.append(f"Upstream source directory does not exist: {root}")
    if missing_files:
        errors.append("Missing upstream source files: " + ", ".join(missing_files))

    missing_text: dict[str, list[str]] = {}
    for name, markers in (required_text or {}).items():
        try:
            source = (root / name).read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            missing_text[name] = list(markers)
        else:
            absent = [marker for marker in markers if marker not in source]
            if absent:
                missing_text[name] = absent
    if missing_text:
        errors.append("Required upstream compatibility patch is missing or incomplete")

    revision = None
    dirty = None
    provenance = "unverified_archive"
    manifest = root / "UPSTREAM.json"
    hashes_verified = False
    hash_errors: list[str] = []
    metadata_revision = None
    if manifest.is_file():
        provenance = "vendored"
        try:
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            if not isinstance(metadata, Mapping):
                raise ValueError("UPSTREAM.json must contain an object")
            metadata_revision = metadata.get("revision")
            if not isinstance(metadata_revision, str) or not metadata_revision:
                raise ValueError("UPSTREAM.json must declare its revision")
            inventory = metadata.get("files")
            if not isinstance(inventory, Mapping) or not inventory:
                raise ValueError("UPSTREAM.json must contain a nonempty files/hash mapping")
            for name in required_files:
                if name not in inventory:
                    hash_errors.append(f"{name}: missing from source inventory")
            for name, expected_hash in inventory.items():
                relative = Path(name)
                path = root / relative
                if (relative.is_absolute() or ".." in relative.parts
                        or not path.resolve().is_relative_to(root)):
                    hash_errors.append(f"{name}: source inventory path is outside its root")
                    continue
                if (not isinstance(expected_hash, str) or len(expected_hash) != 64
                        or any(character not in "0123456789abcdef" for character in expected_hash)):
                    hash_errors.append(f"{name}: invalid SHA-256 in source inventory")
                    continue
                try:
                    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                except OSError:
                    hash_errors.append(f"{name}: missing or unreadable source file")
                else:
                    if actual_hash != expected_hash:
                        hash_errors.append(f"{name}: source SHA-256 differs from inventory")
            hashes_verified = not hash_errors
            revision = metadata_revision
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            errors.append(f"Invalid upstream source inventory: {exc}")
        if hash_errors:
            errors.append("Vendored upstream source does not match its file inventory")
    if (root / ".git").exists():
        provenance = "git"
        try:
            result = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=False, timeout=5,
            )
            if result.returncode == 0:
                revision = result.stdout.strip() or None
            changes = subprocess.run(
                ["git", "-C", str(root), "diff", "--quiet", "HEAD", "--"],
                capture_output=True, check=False, timeout=5,
            )
            if changes.returncode in {0, 1}:
                dirty = changes.returncode == 1
        except (OSError, subprocess.TimeoutExpired):
            pass
        if metadata_revision and revision and metadata_revision != revision:
            errors.append("UPSTREAM.json revision differs from the source Git revision")
    revision_matches = revision == expected_revision if revision and expected_revision else None
    if revision_matches is False:
        errors.append(f"Upstream revision {revision} differs from expected {expected_revision}")

    return {
        "source_root": str(root),
        "source_required": True,
        "source_available": source_available,
        "missing_source_files": missing_files,
        "missing_source_text": missing_text,
        "source_git_revision": revision,
        "expected_source_git_revision": expected_revision,
        "source_revision_matches": revision_matches,
        "source_revision_verified": revision_matches is True and (provenance == "git" or hashes_verified),
        "source_git_dirty": dirty,
        "source_provenance": provenance,
        "source_manifest": str(manifest) if manifest.is_file() else None,
        "source_file_hashes_verified": hashes_verified,
        "source_hash_errors": hash_errors,
        "source_errors": errors,
        "source_ready": source_available and not errors,
    }
