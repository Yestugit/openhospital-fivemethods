from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def atomic_write_json(path: str | Path, value: Any) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    # A same-directory temporary file keeps replacement atomic. A short name
    # avoids crossing Windows' path limit when the target itself still fits.
    fd, temporary = tempfile.mkstemp(prefix=".tmp-", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Windows indexers/antivirus can briefly hold a file without share-delete.
        # Retry the atomic replacement; never delete the previous checkpoint.
        for attempt in range(12):
            try:
                os.replace(temporary, target)
                break
            except OSError as exc:
                if getattr(exc, "winerror", None) not in {5, 32} or attempt == 11:
                    raise
                import time
                time.sleep(min(1.0, 0.05 * (2 ** attempt)))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: str | Path, default: Any) -> Any:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        return default
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read method storage {target}: {exc}") from exc


def storage_bytes(path: str | Path | Iterable[str | Path]) -> int:
    """Measure live method data, excluding telemetry and recovery copies."""
    values = [path] if isinstance(path, (str, Path)) else list(path)
    files: set[Path] = set()
    excluded_dirs = {"snapshots", "state-snapshots", "adapter_snapshots", "checkpoints", "logs", "__pycache__", ".git"}
    telemetry_files = {"memory-run.json", "usage.jsonl", "operations.jsonl", "usage_records.json",
                       "memory_traces.jsonl", "accounting.json", "episodes.jsonl", "adapter_state.json",
                       "snapshot_meta.json"}
    for value in values:
        root = Path(value).expanduser().resolve()
        # Chroma creates HNSW metadata after its first persistence threshold.
        # Its filename can exceed MAX_PATH even when the database folder fits.
        raw = str(root)
        if os.name == "nt" and not raw.startswith("\\\\?\\"):
            root = Path("\\\\?\\UNC\\" + raw[2:] if raw.startswith("\\\\") else "\\\\?\\" + raw)
        if any(part.lower() in excluded_dirs for part in root.parts):
            continue
        if root.is_file():
            files.add(root)
        elif root.exists():
            for directory, folders, names in os.walk(root):
                folders[:] = [name for name in folders if name.lower() not in excluded_dirs
                              and not (Path(directory) / name).is_symlink()]
                files.update((Path(directory) / name).resolve() for name in names)
    return sum(file.stat().st_size for file in files
               if file.name not in telemetry_files and file.suffix.lower() not in {".log", ".pyc"})

