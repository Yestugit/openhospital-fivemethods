"""Bearer-token checks for the unified contest service."""

from __future__ import annotations

import hashlib
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

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


@dataclass(frozen=True)
class ServiceAuthContext:
    dataset_key: str
    account: str = ""
    mode: str = ""
    issued_for: str = ""
    token_hash: str = ""
    bearer_authenticated: bool = False


class ServiceAuth:
    def __init__(
        self,
        *,
        pg_dsn: str | None = None,
    ):
        self.pg_dsn = str(pg_dsn or "").strip()
        self._token_table_ready = False
        self._token_table_lock = threading.Lock()
        self._pool: Any | None = None

    def context_for_credentials(self, *, bearer_token: str | None = None) -> ServiceAuthContext:
        token_context = self.context_for_bearer_token(bearer_token)
        if token_context:
            return token_context
        raise PermissionError("X-Hospital-Service-Token is required.")

    def context_for_bearer_token(self, bearer_token: str | None) -> ServiceAuthContext | None:
        token = str(bearer_token or "").strip()
        if not token:
            return None
        if not self.pg_dsn:
            raise PermissionError("Service bearer token validation is not configured.")

        self._ensure_token_table()
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT token_hash, account, mode, dataset_key, issued_for
                    FROM contest_service_access_tokens
                    WHERE token_hash = %s
                      AND revoked_at IS NULL
                      AND (expires_at IS NULL OR expires_at > now())
                    """,
                    (token_hash,),
                )
                row = cur.fetchone()
                if row:
                    self._require_competition_access(cur, row)
        if not row:
            raise PermissionError("Invalid or expired service token.")
        dataset_key = str(row.get("dataset_key") or "").strip()
        if not dataset_key:
            raise PermissionError("Invalid service token dataset.")
        return ServiceAuthContext(
            dataset_key=dataset_key,
            account=str(row.get("account") or "").strip(),
            mode=str(row.get("mode") or "").strip(),
            issued_for=str(row.get("issued_for") or "").strip(),
            token_hash=str(row.get("token_hash") or "").strip(),
            bearer_authenticated=True,
        )

    def _require_competition_access(self, cur: Any, token: dict[str, Any]) -> None:
        # Internal scoring can finish or re-evaluate archived submissions.
        if token.get("mode") == "evaluation":
            return
        cur.execute("SELECT track FROM contest_users WHERE account = %s", (token["account"],))
        user = cur.fetchone()
        if not user:
            raise PermissionError("服务令牌对应的参赛账号不存在。")
        if user["track"] == "carbon":
            raise PermissionError("碳基训练与测试已关闭，原密钥仅可用于网页登录。")
        cur.execute("SELECT value FROM contest_runtime_settings WHERE key = 'competition_round'")
        row = cur.fetchone()
        state = (row or {}).get("value") or {}
        if state.get("maintenance"):
            raise PermissionError("正在切换复赛，训练与测试暂时关闭。")
        if state.get("round") != "semifinal":
            return
        cur.execute("""
            SELECT 1 FROM competition_qualifiers q
            JOIN contest_users u ON u.account = q.account AND u.track = 'silicon'
            WHERE q.competition_round = 'semifinal' AND q.account = %s
        """, (token["account"],))
        if not cur.fetchone():
            raise PermissionError("仅晋级复赛的硅基选手可以训练和测试，当前训练密钥不可用于训练。")
        expected_dataset = "train" if token.get("mode") == "train" else "advanced_test"
        if token.get("dataset_key") != expected_dataset:
            raise PermissionError("该服务令牌不属于当前复赛题库。")

    def _ensure_token_table(self) -> None:
        if self._token_table_ready:
            return
        with self._token_table_lock:
            if self._token_table_ready:
                return
            with self._connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS contest_service_access_tokens (
                            token_hash TEXT PRIMARY KEY,
                            token_value TEXT,
                            account TEXT NOT NULL,
                            mode TEXT NOT NULL CHECK (mode IN ('train', 'test', 'evaluation')),
                            dataset_key TEXT NOT NULL,
                            issued_for TEXT NOT NULL,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                            expires_at TIMESTAMPTZ,
                            revoked_at TIMESTAMPTZ,
                            revoked_reason TEXT NOT NULL DEFAULT '',
                            metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                        )
                        """
                    )
                    cur.execute("ALTER TABLE contest_service_access_tokens ADD COLUMN IF NOT EXISTS token_value TEXT")
                    cur.execute("ALTER TABLE contest_service_access_tokens ALTER COLUMN expires_at DROP NOT NULL")
                    cur.execute(
                        """
                        ALTER TABLE contest_service_access_tokens
                        DROP CONSTRAINT IF EXISTS contest_service_access_tokens_mode_check
                        """
                    )
                    cur.execute(
                        """
                        ALTER TABLE contest_service_access_tokens
                        ADD CONSTRAINT contest_service_access_tokens_mode_check
                        CHECK (mode IN ('train', 'test', 'evaluation'))
                        """
                    )
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_service_access_tokens_active
                        ON contest_service_access_tokens (account, mode, issued_for, expires_at)
                        WHERE revoked_at IS NULL
                        """
                    )
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS contest_runtime_settings (
                            key TEXT PRIMARY KEY,
                            value JSONB NOT NULL DEFAULT '{}'::jsonb,
                            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                        )
                    """)
            self._token_table_ready = True

    @contextmanager
    def _connection(self):
        if psycopg is None:
            raise RuntimeError("psycopg is required for service bearer token validation.") from _PSYCOPG_IMPORT_ERROR

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


def _pool_max_size() -> int:
    return load_settings().db_pool_max_size
