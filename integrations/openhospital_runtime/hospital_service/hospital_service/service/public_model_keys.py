from __future__ import annotations

import asyncio
import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

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


PUBLIC_MODEL_KEY_SETTING = "public_model_key_scheduler"


@dataclass(frozen=True)
class PublicModelKey:
    key_id: str
    label: str
    api_key: str
    priority: int


class PublicModelKeyPool:
    def __init__(self, pg_dsn: str):
        self.pg_dsn = pg_dsn
        self._pool: Any | None = None
        self.init_db()

    @contextmanager
    def _connection(self):
        if psycopg is None:
            raise RuntimeError("psycopg is required for public model API key pooling.") from _PSYCOPG_IMPORT_ERROR
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
                    CREATE TABLE IF NOT EXISTS contest_runtime_settings (
                        key TEXT PRIMARY KEY,
                        value JSONB NOT NULL DEFAULT '{}'::jsonb,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS contest_public_model_api_keys (
                        key_id TEXT PRIMARY KEY,
                        label TEXT NOT NULL,
                        api_key TEXT NOT NULL,
                        enabled BOOLEAN NOT NULL DEFAULT true,
                        selected BOOLEAN NOT NULL DEFAULT true,
                        priority INTEGER NOT NULL DEFAULT 100,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS contest_public_model_key_usage (
                        key_id TEXT PRIMARY KEY REFERENCES contest_public_model_api_keys(key_id) ON DELETE CASCADE,
                        requests BIGINT NOT NULL DEFAULT 0,
                        input_tokens BIGINT NOT NULL DEFAULT 0,
                        output_tokens BIGINT NOT NULL DEFAULT 0,
                        total_tokens BIGINT NOT NULL DEFAULT 0,
                        failures BIGINT NOT NULL DEFAULT 0,
                        last_used_at TIMESTAMPTZ,
                        last_error_at TIMESTAMPTZ,
                        last_error TEXT NOT NULL DEFAULT '',
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )

    def selected_keys(self) -> list[PublicModelKey]:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT key_id, label, api_key, priority
                    FROM contest_public_model_api_keys
                    WHERE enabled = true
                      AND selected = true
                    ORDER BY priority ASC, created_at ASC
                    """
                )
                rows = cur.fetchall()
        return [
            PublicModelKey(
                key_id=str(row.get("key_id") or ""),
                label=str(row.get("label") or ""),
                api_key=str(row.get("api_key") or ""),
                priority=safe_int(row.get("priority"), 100),
            )
            for row in rows
            if str(row.get("key_id") or "").strip() and str(row.get("api_key") or "").strip()
        ]

    def scheduler_mode(self) -> str:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT value FROM contest_runtime_settings WHERE key = %s",
                    (PUBLIC_MODEL_KEY_SETTING,),
                )
                row = cur.fetchone()
        value = row.get("value") if isinstance(row, dict) else {}
        mode = str(value.get("mode") if isinstance(value, dict) else "priority").strip().lower()
        return mode if mode in {"priority", "random"} else "priority"

    def run(
        self,
        *,
        operation: Callable[[str], Any],
        usage_extractor: Callable[[Any], dict[str, int]],
    ) -> Any:
        keys = self.selected_keys()
        if not keys:
            raise LookupError("no public model API keys configured")
        last_error: Exception | None = None
        for item in self._attempt_order(keys):
            try:
                result = operation(item.api_key)
                self.record_usage(item.key_id, usage_extractor(result))
                return result
            except Exception as exc:  # noqa: BLE001 - fail over to the next public key.
                last_error = exc
                self.record_failure(item.key_id, exc)
        raise RuntimeError(f"All public model API keys failed: {last_error}") from last_error

    async def run_async(
        self,
        *,
        operation: Callable[[str], Awaitable[Any]],
        usage_extractor: Callable[[Any], dict[str, int]],
    ) -> Any:
        keys = await asyncio.to_thread(self.selected_keys)
        if not keys:
            raise LookupError("no public model API keys configured")
        last_error: Exception | None = None
        for item in await asyncio.to_thread(self._attempt_order, keys):
            try:
                result = await operation(item.api_key)
                await asyncio.to_thread(self.record_usage, item.key_id, usage_extractor(result))
                return result
            except Exception as exc:  # noqa: BLE001 - fail over to the next public key.
                last_error = exc
                await asyncio.to_thread(self.record_failure, item.key_id, exc)
        raise RuntimeError(f"All public model API keys failed: {last_error}") from last_error

    def _attempt_order(self, keys: list[PublicModelKey]) -> list[PublicModelKey]:
        if self.scheduler_mode() == "random":
            ordered = list(keys)
            random.shuffle(ordered)
            return ordered
        return sorted(keys, key=lambda item: (item.priority, item.label, item.key_id))

    def record_usage(self, key_id: str, usage: dict[str, int]) -> None:
        normalized = normalize_usage(usage)
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO contest_public_model_key_usage (
                        key_id, requests, input_tokens, output_tokens, total_tokens, last_used_at
                    )
                    VALUES (%s, 1, %s, %s, %s, now())
                    ON CONFLICT (key_id) DO UPDATE SET
                        requests = contest_public_model_key_usage.requests + 1,
                        input_tokens = contest_public_model_key_usage.input_tokens + EXCLUDED.input_tokens,
                        output_tokens = contest_public_model_key_usage.output_tokens + EXCLUDED.output_tokens,
                        total_tokens = contest_public_model_key_usage.total_tokens + EXCLUDED.total_tokens,
                        last_used_at = now(),
                        updated_at = now()
                    """,
                    (
                        key_id,
                        normalized["prompt_tokens"],
                        normalized["completion_tokens"],
                        normalized["total_tokens"],
                    ),
                )

    def record_failure(self, key_id: str, error: Exception) -> None:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO contest_public_model_key_usage (
                        key_id, failures, last_error_at, last_error
                    )
                    VALUES (%s, 1, now(), %s)
                    ON CONFLICT (key_id) DO UPDATE SET
                        failures = contest_public_model_key_usage.failures + 1,
                        last_error_at = now(),
                        last_error = EXCLUDED.last_error,
                        updated_at = now()
                    """,
                    (key_id, str(error)[:1000]),
                )

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()


def normalize_usage(value: dict[str, int] | None) -> dict[str, int]:
    value = value if isinstance(value, dict) else {}
    prompt_tokens = safe_int(value.get("prompt_tokens") or value.get("input_tokens") or value.get("inputTokens"), 0)
    completion_tokens = safe_int(
        value.get("completion_tokens") or value.get("output_tokens") or value.get("outputTokens"),
        0,
    )
    total_tokens = safe_int(value.get("total_tokens") or value.get("totalTokens"), prompt_tokens + completion_tokens)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _pool_max_size() -> int:
    return load_settings().db_pool_max_size
