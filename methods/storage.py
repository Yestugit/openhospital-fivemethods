from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import hashlib
from threading import RLock
from pathlib import Path
from typing import Any, Mapping

from mematlas.core.plugin import MethodServices
from mematlas.core.files import atomic_write_json, read_json


def method_data_dir(
    services: MethodServices,
    method_id: str,
    configured: str | Path | None = None,
) -> Path:
    """Return a stable, method-owned directory for embedded state.

    Explicit paths are respected so benchmark runs can isolate methods. The
    default lives below the run artifact directory and therefore remains
    portable across machines and subprocess executors.
    """
    if configured:
        root = Path(configured).expanduser()
    elif Path(services.artifact_dir) == Path(".mematlas/artifacts"):
        # Core classes are also useful in isolation (for unit tests and
        # notebooks). Their implicit state must not leak between instances.
        root = Path(tempfile.mkdtemp(prefix=f"mematlas-{method_id}-"))
    else:
        root = Path(services.artifact_dir) / "methods" / method_id
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return services.register_resource(root)


def configured_file(
    services: MethodServices,
    method_id: str,
    configured: str | Path | None,
    default_name: str,
) -> Path:
    """Resolve a file option while accepting either a file or a directory."""
    if configured:
        path = Path(configured).expanduser()
        if path.suffix or path.exists() and path.is_file():
            path = path.resolve()
        else:
            path = (path / default_name).resolve()
    else:
        path = method_data_dir(services, method_id) / default_name
    path.parent.mkdir(parents=True, exist_ok=True)
    return services.register_resource(path)


def ascii_database_path(anchor: str | Path, filename: str, *, services: MethodServices | None = None) -> Path:
    """Return a stable ASCII path for native Windows database engines."""
    value = str(Path(anchor).expanduser().resolve())
    anchor_path = Path(value)
    if os.name != "nt" or value.isascii():
        path = (anchor_path if anchor_path.is_dir() else anchor_path.parent) / filename
        return services.register_resource(path) if services is not None else path
    root = Path(tempfile.gettempdir()) / "mematlas-native-db"
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]
    path = root / f"{digest}-{filename}"
    return services.register_resource(path) if services is not None else path






class SQLiteState:
    """Small JSON-payload store used by embedded method backends.

    Keeping payloads method-defined lets each adapter preserve its native data
    model while SQLite supplies durable transactions and restart semantics.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self._lock = RLock()
        with self._lock:
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            self.connection.commit()

    def load(self, key: str, default: Any) -> Any:
        with self._lock:
            row = self.connection.execute(
                "SELECT payload FROM state WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row[0])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON payload in {self.path} for {key!r}") from exc

    def save(self, key: str, value: Any) -> None:
        payload = json.dumps(value, ensure_ascii=False, default=str)
        with self._lock:
            self.connection.execute(
                "INSERT INTO state(key, payload) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET payload=excluded.payload",
                (key, payload),
            )
            self.connection.commit()

    def clear(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self.connection.execute("DELETE FROM state")
            else:
                self.connection.execute("DELETE FROM state WHERE key = ?", (key,))
            self.connection.commit()

    def close(self) -> None:
        with self._lock:
            self.connection.close()


def load_mapping(store: SQLiteState, key: str) -> dict[str, Any]:
    value = store.load(key, {})
    return dict(value) if isinstance(value, Mapping) else {}
