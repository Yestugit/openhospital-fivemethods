"""Persistent shared storage for patient examination results."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Protocol

try:
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool
except ModuleNotFoundError as exc:
    dict_row = None  # type: ignore[assignment]
    ConnectionPool = None  # type: ignore[assignment]
    _PSYCOPG_IMPORT_ERROR = exc
else:
    _PSYCOPG_IMPORT_ERROR = None

BinaryStatus = Literal["normal", "abnormal"]
ResultSource = Literal["existing", "generated"]


@dataclass(frozen=True)
class ExaminationResultRecord:
    dataset_key: str
    patient_id: str
    examination_name: str
    result: Any
    indicator_statuses: dict[str, BinaryStatus]
    status: BinaryStatus
    abnormal_indicators: tuple[str, ...]
    source: ResultSource
    model_name: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"normal", "abnormal"}:
            raise ValueError(f"Invalid examination status: {self.status}")
        if self.source not in {"existing", "generated"}:
            raise ValueError(f"Invalid examination result source: {self.source}")
        if self.source == "generated" and not str(self.model_name or "").strip():
            raise ValueError("Generated examination result must include model_name")
        invalid = sorted(
            name for name, status in self.indicator_statuses.items()
            if status not in {"normal", "abnormal"}
        )
        if invalid:
            raise ValueError(f"Invalid indicator status for: {', '.join(invalid)}")


class ExaminationResultStore(Protocol):
    def get(
        self,
        dataset_key: str,
        patient_id: str,
        examination_name: str,
    ) -> ExaminationResultRecord | None: ...

    def list_patient(
        self,
        dataset_key: str,
        patient_id: str,
    ) -> dict[str, ExaminationResultRecord]: ...

    def put_if_absent(self, record: ExaminationResultRecord) -> ExaminationResultRecord: ...


class InMemoryExaminationResultStore:
    def __init__(self) -> None:
        self._records: dict[tuple[str, str, str], ExaminationResultRecord] = {}
        self._lock = threading.RLock()

    def get(
        self,
        dataset_key: str,
        patient_id: str,
        examination_name: str,
    ) -> ExaminationResultRecord | None:
        with self._lock:
            return _copy_record(self._records.get((dataset_key, patient_id, examination_name)))

    def list_patient(
        self,
        dataset_key: str,
        patient_id: str,
    ) -> dict[str, ExaminationResultRecord]:
        with self._lock:
            return {
                examination_name: _copy_record(record)
                for (record_dataset, record_patient, examination_name), record in self._records.items()
                if record_dataset == dataset_key and record_patient == patient_id
            }

    def put_if_absent(self, record: ExaminationResultRecord) -> ExaminationResultRecord:
        key = (record.dataset_key, record.patient_id, record.examination_name)
        with self._lock:
            stored = self._records.setdefault(key, _copy_record(record))
            copied = _copy_record(stored)
            assert copied is not None
            return copied

    def close(self) -> None:
        return None


class PostgresExaminationResultStore:
    def __init__(self, pg_dsn: str, *, pool_max_size: int = 10):
        if ConnectionPool is None:
            raise RuntimeError("psycopg-pool is required for examination result storage") from _PSYCOPG_IMPORT_ERROR
        self._pool = ConnectionPool(
            conninfo=pg_dsn,
            kwargs={"row_factory": dict_row},
            min_size=1,
            max_size=max(1, int(pool_max_size)),
            open=False,
        )
        self._pool.open()
        self.init_db()

    @contextmanager
    def _connection(self):
        with self._pool.connection() as conn:
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def init_db(self) -> None:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS contest_patient_examination_results (
                        dataset_key TEXT NOT NULL,
                        patient_id TEXT NOT NULL,
                        examination_name TEXT NOT NULL,
                        result JSONB NOT NULL,
                        indicator_statuses JSONB NOT NULL DEFAULT '{}'::jsonb,
                        status TEXT NOT NULL CHECK (status IN ('normal', 'abnormal')),
                        abnormal_indicators JSONB NOT NULL DEFAULT '[]'::jsonb,
                        source TEXT NOT NULL CHECK (source IN ('existing', 'generated')),
                        model_name TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        PRIMARY KEY (dataset_key, patient_id, examination_name)
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_patient_examination_results_patient
                    ON contest_patient_examination_results (dataset_key, patient_id)
                    """
                )

    def get(
        self,
        dataset_key: str,
        patient_id: str,
        examination_name: str,
    ) -> ExaminationResultRecord | None:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT dataset_key, patient_id, examination_name, result,
                           indicator_statuses, status, abnormal_indicators, source, model_name
                    FROM contest_patient_examination_results
                    WHERE dataset_key = %s AND patient_id = %s AND examination_name = %s
                    """,
                    (dataset_key, patient_id, examination_name),
                )
                row = cur.fetchone()
        return _record_from_row(row) if row is not None else None

    def list_patient(
        self,
        dataset_key: str,
        patient_id: str,
    ) -> dict[str, ExaminationResultRecord]:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT dataset_key, patient_id, examination_name, result,
                           indicator_statuses, status, abnormal_indicators, source, model_name
                    FROM contest_patient_examination_results
                    WHERE dataset_key = %s AND patient_id = %s
                    ORDER BY examination_name
                    """,
                    (dataset_key, patient_id),
                )
                rows = cur.fetchall()
        return {
            record.examination_name: record
            for row in rows
            if (record := _record_from_row(row)) is not None
        }

    def put_if_absent(self, record: ExaminationResultRecord) -> ExaminationResultRecord:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO contest_patient_examination_results (
                        dataset_key, patient_id, examination_name, result,
                        indicator_statuses, status, abnormal_indicators, source, model_name
                    )
                    VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s::jsonb, %s, %s)
                    ON CONFLICT (dataset_key, patient_id, examination_name) DO NOTHING
                    """,
                    (
                        record.dataset_key,
                        record.patient_id,
                        record.examination_name,
                        json.dumps(record.result, ensure_ascii=False),
                        json.dumps(record.indicator_statuses, ensure_ascii=False),
                        record.status,
                        json.dumps(record.abnormal_indicators, ensure_ascii=False),
                        record.source,
                        record.model_name,
                    ),
                )
                cur.execute(
                    """
                    SELECT dataset_key, patient_id, examination_name, result,
                           indicator_statuses, status, abnormal_indicators, source, model_name
                    FROM contest_patient_examination_results
                    WHERE dataset_key = %s AND patient_id = %s AND examination_name = %s
                    """,
                    (record.dataset_key, record.patient_id, record.examination_name),
                )
                row = cur.fetchone()
        stored = _record_from_row(row)
        if stored is None:
            raise RuntimeError("Examination result insert completed without a stored row")
        return stored

    def close(self) -> None:
        self._pool.close()


def _record_from_row(row: Any) -> ExaminationResultRecord | None:
    if not isinstance(row, dict):
        return None
    indicator_statuses = row.get("indicator_statuses")
    abnormal_indicators = row.get("abnormal_indicators")
    return ExaminationResultRecord(
        dataset_key=str(row.get("dataset_key") or ""),
        patient_id=str(row.get("patient_id") or ""),
        examination_name=str(row.get("examination_name") or ""),
        result=deepcopy(row.get("result")),
        indicator_statuses={
            str(name): str(status)  # type: ignore[misc]
            for name, status in (indicator_statuses.items() if isinstance(indicator_statuses, dict) else ())
        },
        status=str(row.get("status") or ""),  # type: ignore[arg-type]
        abnormal_indicators=tuple(
            str(name) for name in abnormal_indicators
        ) if isinstance(abnormal_indicators, list) else (),
        source=str(row.get("source") or ""),  # type: ignore[arg-type]
        model_name=str(row.get("model_name")) if row.get("model_name") is not None else None,
    )


def _copy_record(record: ExaminationResultRecord | None) -> ExaminationResultRecord | None:
    if record is None:
        return None
    return ExaminationResultRecord(
        dataset_key=record.dataset_key,
        patient_id=record.patient_id,
        examination_name=record.examination_name,
        result=deepcopy(record.result),
        indicator_statuses=dict(record.indicator_statuses),
        status=record.status,
        abnormal_indicators=tuple(record.abnormal_indicators),
        source=record.source,
        model_name=record.model_name,
    )
