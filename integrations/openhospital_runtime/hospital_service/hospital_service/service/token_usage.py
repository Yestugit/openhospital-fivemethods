"""PostgreSQL-backed token usage persistence for the contest service."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import atexit
import threading
import time
from typing import Any, Protocol

try:
    from .config import load_settings
except ImportError:
    from config import load_settings

try:
    import psycopg
    from psycopg.rows import dict_row
except ModuleNotFoundError as exc:
    psycopg = None  # type: ignore[assignment]
    dict_row = None  # type: ignore[assignment]
    _PSYCOPG_IMPORT_ERROR = exc
else:
    _PSYCOPG_IMPORT_ERROR = None

try:
    from psycopg_pool import ConnectionPool
except ModuleNotFoundError:
    ConnectionPool = None  # type: ignore[assignment]


USAGE_KEYS = ("requests", "input_tokens", "output_tokens", "total_tokens")
DEFAULT_FLUSH_INTERVAL_SECONDS = 2.0
DEFAULT_FLUSH_THRESHOLD = 20
class TokenUsageStore(Protocol):
    def get_usage(self, *, dataset_key: str, team_id: str) -> dict[str, int]:
        """Return aggregate token usage for one team in one dataset."""

    def list_usage(self, *, dataset_key: str) -> dict[str, dict[str, int]]:
        """Return aggregate token usage for every team in one dataset."""

    def increment_usage(
        self,
        *,
        dataset_key: str,
        team_id: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
    ) -> dict[str, int]:
        """Atomically increment and return aggregate token usage."""


class PostgresTokenUsageStore:
    def __init__(self, pg_dsn: str):
        self.pg_dsn = pg_dsn
        self._pool: Any | None = None
        self._pending_usage: dict[tuple[str, str], dict[str, int]] = {}
        self._usage_totals: dict[tuple[str, str], dict[str, int]] = {}
        self._pending_lock = threading.Lock()
        self._last_flush_at = time.monotonic()
        self._flush_interval_seconds = DEFAULT_FLUSH_INTERVAL_SECONDS
        self._flush_threshold = DEFAULT_FLUSH_THRESHOLD
        self.init_db()
        atexit.register(self.flush)

    @contextmanager
    def _connection(self):
        if psycopg is None:
            raise RuntimeError("psycopg is required for PostgreSQL token usage storage.") from _PSYCOPG_IMPORT_ERROR

        pool = self._connection_pool()
        if pool is not None:
            with pool.connection() as conn:
                try:
                    yield conn
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
            return

        conn = psycopg.connect(self.pg_dsn, row_factory=dict_row)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _connection_pool(self) -> Any | None:
        if ConnectionPool is None:
            return None
        if self._pool is None:
            self._pool = ConnectionPool(
                conninfo=self.pg_dsn,
                kwargs={"row_factory": dict_row},
                min_size=1,
                max_size=_pool_max_size(),
                open=False,
            )
            self._pool.open()
        return self._pool

    def init_db(self) -> None:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS contest_service_token_usage (
                        dataset_key TEXT NOT NULL,
                        team_id TEXT NOT NULL,
                        requests BIGINT NOT NULL DEFAULT 0,
                        input_tokens BIGINT NOT NULL DEFAULT 0,
                        output_tokens BIGINT NOT NULL DEFAULT 0,
                        total_tokens BIGINT NOT NULL DEFAULT 0,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        PRIMARY KEY (dataset_key, team_id)
                    )
                    """
                )

    def get_usage(self, *, dataset_key: str, team_id: str) -> dict[str, int]:
        self.flush()
        key = (dataset_key, team_id)
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT requests, input_tokens, output_tokens, total_tokens
                    FROM contest_service_token_usage
                    WHERE dataset_key = %s AND team_id = %s
                    """,
                    (dataset_key, team_id),
                )
                row = cur.fetchone()
        usage = normalize_usage_record(row) if row else new_usage_record()
        with self._pending_lock:
            self._usage_totals[key] = dict(usage)
        return usage

    def list_usage(self, *, dataset_key: str) -> dict[str, dict[str, int]]:
        self.flush()
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT team_id, requests, input_tokens, output_tokens, total_tokens
                    FROM contest_service_token_usage
                    WHERE dataset_key = %s
                    ORDER BY team_id
                    """,
                    (dataset_key,),
                )
                rows = cur.fetchall()
        return {
            str(row["team_id"]): normalize_usage_record(row)
            for row in rows
        }

    def increment_usage(
        self,
        *,
        dataset_key: str,
        team_id: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
    ) -> dict[str, int]:
        key = (dataset_key, team_id)
        delta = {
            "requests": 1,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": int(total_tokens),
        }
        self._ensure_usage_total_loaded(dataset_key=dataset_key, team_id=team_id)
        with self._pending_lock:
            usage = self._pending_usage.setdefault(key, new_usage_record())
            total = self._usage_totals.setdefault(key, new_usage_record())
            for usage_key in USAGE_KEYS:
                usage[usage_key] += delta[usage_key]
                total[usage_key] += delta[usage_key]
            snapshot = dict(total)
            pending_size = len(self._pending_usage)
            should_flush = (
                pending_size >= self._flush_threshold
                or time.monotonic() - self._last_flush_at >= self._flush_interval_seconds
            )
        if should_flush:
            self.flush()
        return snapshot

    def flush(self) -> None:
        pending = self._drain_pending_usage()
        if not pending:
            return
        try:
            self._write_usage_batch(pending)
        except Exception:
            self._restore_pending_usage(pending)
            raise

    def close(self) -> None:
        self.flush()

    def _drain_pending_usage(self) -> dict[tuple[str, str], dict[str, int]]:
        with self._pending_lock:
            pending = self._pending_usage
            self._pending_usage = {}
            self._last_flush_at = time.monotonic()
        return pending

    def _restore_pending_usage(self, pending: dict[tuple[str, str], dict[str, int]]) -> None:
        with self._pending_lock:
            for key, usage in pending.items():
                restored = self._pending_usage.setdefault(key, new_usage_record())
                for usage_key in USAGE_KEYS:
                    restored[usage_key] += int(usage.get(usage_key) or 0)

    def _persisted_usage(self, *, dataset_key: str, team_id: str) -> dict[str, int]:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT requests, input_tokens, output_tokens, total_tokens
                    FROM contest_service_token_usage
                    WHERE dataset_key = %s AND team_id = %s
                    """,
                    (dataset_key, team_id),
                )
                row = cur.fetchone()
        return normalize_usage_record(row) if row else new_usage_record()

    def _ensure_usage_total_loaded(self, *, dataset_key: str, team_id: str) -> None:
        key = (dataset_key, team_id)
        with self._pending_lock:
            if key in self._usage_totals:
                return
        persisted = self._persisted_usage(dataset_key=dataset_key, team_id=team_id)
        with self._pending_lock:
            self._usage_totals.setdefault(key, persisted)

    def _write_usage_batch(self, pending: dict[tuple[str, str], dict[str, int]]) -> None:
        with self._connection() as conn:
            with conn.cursor() as cur:
                for (dataset_key, team_id), usage in pending.items():
                    cur.execute(
                        """
                        INSERT INTO contest_service_token_usage (
                            dataset_key,
                            team_id,
                            requests,
                            input_tokens,
                            output_tokens,
                            total_tokens
                        )
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (dataset_key, team_id) DO UPDATE SET
                            requests = contest_service_token_usage.requests + EXCLUDED.requests,
                            input_tokens = contest_service_token_usage.input_tokens + EXCLUDED.input_tokens,
                            output_tokens = contest_service_token_usage.output_tokens + EXCLUDED.output_tokens,
                            total_tokens = contest_service_token_usage.total_tokens + EXCLUDED.total_tokens,
                            updated_at = now()
                        """,
                        (
                            dataset_key,
                            team_id,
                            int(usage.get("requests") or 0),
                            int(usage.get("input_tokens") or 0),
                            int(usage.get("output_tokens") or 0),
                            int(usage.get("total_tokens") or 0),
                        ),
                    )


class InMemoryTokenUsageStore:
    """Non-persistent fallback for direct unit-level construction."""

    def __init__(self):
        self._usage_by_dataset: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(new_usage_record)
        )

    def get_usage(self, *, dataset_key: str, team_id: str) -> dict[str, int]:
        return dict(self._usage_by_dataset[dataset_key].get(team_id, new_usage_record()))

    def list_usage(self, *, dataset_key: str) -> dict[str, dict[str, int]]:
        return {
            key: dict(value)
            for key, value in sorted(self._usage_by_dataset[dataset_key].items())
        }

    def increment_usage(
        self,
        *,
        dataset_key: str,
        team_id: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
    ) -> dict[str, int]:
        usage = self._usage_by_dataset[dataset_key][team_id]
        usage["requests"] += 1
        usage["input_tokens"] += int(input_tokens)
        usage["output_tokens"] += int(output_tokens)
        usage["total_tokens"] += int(total_tokens)
        return dict(usage)


def new_usage_record() -> dict[str, int]:
    return {
        "requests": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


def normalize_usage_record(usage: dict[str, Any]) -> dict[str, int]:
    record = new_usage_record()
    for key in USAGE_KEYS:
        record[key] = _safe_int(usage.get(key), 0)
    return record


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _pool_max_size() -> int:
    return load_settings().db_pool_max_size
