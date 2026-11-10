from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict, dataclass, field
from dataclasses import replace
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, Mapping
from datetime import datetime, timezone
from uuid import uuid4
import math

from mematlas.core.models import OperationTrace


class ProcessResourceMonitor:
    """Sample RSS for this process and its method-worker descendants."""

    def __init__(self, *, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self._root = None
        self._task: asyncio.Task | None = None
        self._running = False
        self._baseline_rss_bytes: int | None = None
        self._peak_rss_bytes: int | None = None
        self._sample_count = 0
        self._error: str | None = None

    def _total_rss(self) -> int | None:
        if self._root is None:
            return None
        try:
            processes = [self._root, *self._root.children(recursive=True)]
        except Exception:
            processes = [self._root]
        total = 0
        observed = False
        for process in processes:
            try:
                total += int(process.memory_info().rss)
                observed = True
            except Exception:
                continue
        return total if observed else None

    def _sample(self) -> None:
        value = self._total_rss()
        if value is None:
            return
        self._sample_count += 1
        if self._baseline_rss_bytes is None:
            self._baseline_rss_bytes = value
        self._peak_rss_bytes = max(self._peak_rss_bytes or value, value)

    async def _poll(self) -> None:
        while self._running:
            await asyncio.sleep(self.interval_s)
            self._sample()

    async def start(self) -> "ProcessResourceMonitor":
        if self._task is not None:
            return self
        try:
            import psutil
            self._root = psutil.Process(os.getpid())
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            return self
        self._sample()
        self._running = True
        self._task = asyncio.create_task(self._poll())
        return self

    async def stop(self) -> dict[str, int | float | str | None]:
        self._running = False
        if self._task is not None:
            await self._task
        self._sample()
        baseline = self._baseline_rss_bytes
        peak = self._peak_rss_bytes
        return {
            "process_baseline_rss_bytes": baseline,
            "process_peak_rss_bytes": peak,
            "process_peak_rss_mib": peak / (1024 ** 2) if peak is not None else None,
            "process_peak_rss_delta_bytes": (
                max(0, peak - baseline) if peak is not None and baseline is not None else None
            ),
            "process_memory_sample_count": self._sample_count,
            "process_memory_monitor_error": self._error,
        }


@dataclass(frozen=True)
class UsageRecord:
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float | None = None
    metadata: dict = field(default_factory=dict)
    phase: str = "memory"
    usage_known: bool = True
    call_id: str | None = None
    record_id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    stage: str = "unclassified"
    operation: str | None = None
    operation_id: str | None = None
    run_id: str | None = None
    segment_id: str | None = None
    case_id: str | None = None
    resource_kind: str = "llm"
    usage_source: str = "reported"
    status: str = "success"
    currency: str | None = None


def _prompt_cache_summary(records):
    """Reported LLM cache tokens only; missing fields never mean zero hits."""
    llm = [record for record in records if record.resource_kind == "llm"]
    reported, cached, created, inputs = 0, 0, 0, 0
    for record in llm:
        details = record.metadata.get("prompt_tokens_details") or {}
        if not isinstance(details, Mapping):
            continue
        hit = details.get("cached_tokens")
        if (not record.usage_known or not isinstance(hit, int) or isinstance(hit, bool)
                or not 0 <= hit <= record.prompt_tokens):
            continue
        reported += 1
        cached += hit
        inputs += record.prompt_tokens
        write = details.get("cache_creation_input_tokens", details.get("cache_write_tokens"))
        if isinstance(write, int) and not isinstance(write, bool) and 0 <= write <= record.prompt_tokens - hit:
            created += write
    complete = bool(llm) and reported == len(llm)
    return {"cached_prompt_tokens": cached if complete else None,
            "known_cached_prompt_tokens": cached, "known_cache_creation_prompt_tokens": created,
            "cache_reported_calls": reported, "cache_reported_prompt_tokens": inputs,
            "cache_token_coverage": "complete" if complete else "partial" if reported else "unknown",
            "prompt_cache_hit_rate": cached / inputs if inputs else None}


_ACTIVE_USAGE: ContextVar[tuple["UsageLedger", str] | None] = ContextVar("mematlas_usage", default=None)
_ACTIVE_CALL: ContextVar[str | None] = ContextVar("mematlas_usage_call", default=None)
_USAGE_CONTEXT: ContextVar[dict] = ContextVar("mematlas_usage_context", default={})


def current_usage_context() -> dict:
    return dict(_USAGE_CONTEXT.get())


@contextmanager
def usage_context(**values):
    token = _USAGE_CONTEXT.set({**_USAGE_CONTEXT.get(), **values})
    try:
        yield
    finally:
        _USAGE_CONTEXT.reset(token)


@contextmanager
def usage_scope(ledger: "UsageLedger", phase: str):
    token = _ACTIVE_USAGE.set((ledger, phase))
    try:
        yield
    finally:
        _ACTIVE_USAGE.reset(token)


def active_usage() -> tuple["UsageLedger", str] | None:
    return _ACTIVE_USAGE.get()


@dataclass
class UsageLedger:
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    model_pricing: dict[str, dict[str, float]] = field(default_factory=dict)
    currency: str = "USD"
    _records: list[UsageRecord] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock)
    _ids: dict[str, UsageRecord] = field(default_factory=dict)
    _journal: Path | None = None
    _run_id: str | None = None

    def __post_init__(self):
        for price in (self.input_cost_per_million, self.output_cost_per_million,
                      *(value for model in self.model_pricing.values() for key, value in model.items()
                        if key in {"input_cost_per_million", "output_cost_per_million"})):
            if price is not None and (not isinstance(price, (int, float)) or isinstance(price, bool)
                                      or not math.isfinite(price) or price < 0):
                raise ValueError("Model prices must be finite, nonnegative numbers")
        self._ids = {record.record_id: record for record in self._records}

    def add(self, record: UsageRecord, *, imported: bool = False) -> None:
        active = _ACTIVE_USAGE.get()
        if not imported:
            values = current_usage_context()
            values = {key: value for key, value in values.items() if key in {
                "stage", "operation", "operation_id", "segment_id", "case_id", "run_id",
            }}
            record = replace(record, **values)
            if not record.run_id:
                record = replace(record, run_id=self._run_id)
        if active is not None and not imported:
            record = replace(record, phase=active[1])
        if _ACTIVE_CALL.get() is not None and not imported:
            record = replace(record, call_id=_ACTIVE_CALL.get())
        if not record.usage_known:
            record = replace(record, usage_source="unknown")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
               for value in (record.prompt_tokens, record.completion_tokens)) or record.cost is not None and (
               not isinstance(record.cost, (int, float)) or isinstance(record.cost, bool)
               or not math.isfinite(record.cost) or record.cost < 0):
            raise ValueError("Usage and costs must be nonnegative")
        self._append(record)
        if active is not None and active[0] is not self:
            active[0]._append(record)

    def _append(self, record: UsageRecord) -> None:
        with self._lock:
            if record.record_id in self._ids:
                if self._ids[record.record_id] != record:
                    raise ValueError("Conflicting usage records have the same record_id")
                return
            if self._journal is not None:
                with self._journal.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            self._records.append(record)
            self._ids[record.record_id] = record

    def bind_journal(self, path: str | Path, *, run_id: str | None = None) -> None:
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            self.import_journal(target)
        with self._lock:
            self._journal = target
            self._run_id = run_id or self._run_id
            # Rebinding an experiment journal includes setup and restored history.
            # Atomic replacement preserves the previous file on Windows.
            rows = "".join(json.dumps(asdict(row), ensure_ascii=False) + "\n" for row in self._records)
            import tempfile
            fd, name = tempfile.mkstemp(dir=target.parent, prefix=".usage-", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    stream.write(rows)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(name, target)
            finally:
                if os.path.exists(name):
                    os.unlink(name)

    def import_journal(self, path: str | Path) -> None:
        data = Path(path).read_bytes()
        lines = data.splitlines(keepends=True)
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                if index != len(lines) - 1 or line.endswith(b"\n"):
                    raise ValueError("Usage journal is corrupt before its final unfinished record")
                from hashlib import sha256
                self.add(UsageRecord("journal_recovery", "unknown", usage_known=False,
                    record_id="incomplete-" + sha256(data).hexdigest(), timestamp="recovery",
                    status="unknown", metadata={"reason":"Interrupted final usage record", "source":str(path)}),
                    imported=True)
                return
            self.add(UsageRecord(**row), imported=True)

    def cursor(self) -> int:
        return len(self.all())

    def all(self) -> tuple[UsageRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def summary(self, phase: str | tuple[str, ...] | None = None, *, since: int = 0,
                stages: Iterable[str] | None = None, segment_id: str | None = None,
                case_id: str | None = None, operation_id: str | None = None,
                resource_kind: str | None = None) -> dict:
        phases = (phase,) if isinstance(phase, str) else phase
        records = tuple(record for record in self.all()[since:] if (phases is None or record.phase in phases)
                        and (stages is None or record.stage in stages)
                        and (segment_id is None or record.segment_id == segment_id)
                        and (case_id is None or record.case_id == case_id)
                        and (operation_id is None or record.operation_id == operation_id)
                        and (resource_kind is None or record.resource_kind == resource_kind))
        effective_costs: list[float | None] = []
        for record in records:
            cost = record.cost
            pricing = self.model_pricing.get(f"{record.provider}:{record.model}", self.model_pricing.get(record.model, {}))
            if record.currency and record.currency != self.currency:
                raise ValueError("Mixed currencies require an explicit conversion before aggregation")
            input_price = pricing.get("input_cost_per_million", self.input_cost_per_million)
            output_price = pricing.get("output_cost_per_million", self.output_cost_per_million)
            if (
                cost is None
                and record.usage_known
                and input_price is not None
                and output_price is not None
            ):
                cost = (
                    record.prompt_tokens * input_price
                    + record.completion_tokens * output_price
                ) / 1_000_000
            effective_costs.append(cost)
        known_costs = [cost for cost in effective_costs if cost is not None]
        all_costed = len(known_costs) == len(records)
        complete = all(record.usage_known for record in records)
        known_tokens = sum(record.prompt_tokens + record.completion_tokens for record in records)
        return {
            "tracked_calls": len(records),
            **_prompt_cache_summary(records),
            "llm_calls": sum(record.resource_kind == "llm" for record in records),
            "embedding_calls": sum(record.resource_kind == "embedding" for record in records),
            "prompt_tokens": sum(record.prompt_tokens for record in records),
            "completion_tokens": sum(record.completion_tokens for record in records),
            "total_tokens": known_tokens if complete else None,
            "known_total_tokens": known_tokens,
            "costed_calls": len(known_costs),
            "uncosted_calls": len(records) - len(known_costs),
            "known_cost": sum(known_costs),
            "cost": sum(known_costs) if all_costed else None,
            "untracked_calls": sum(not record.usage_known for record in records),
            "token_coverage": "partial" if any(not record.usage_known for record in records) else "complete",
            "estimated_calls": sum(record.usage_source == "estimated" for record in records),
            "failed_calls": sum(record.status != "success" for record in records),
            "currency": self.currency,
        }

    def pricing_spec(self) -> dict:
        return {
            "input_cost_per_million": self.input_cost_per_million,
            "output_cost_per_million": self.output_cost_per_million,
            "model_pricing": dict(self.model_pricing),
            "currency": self.currency,
        }

    def resource_metrics(self) -> dict:
        def measurements(summary):
            values = dict(summary)
            values["known_prompt_tokens"] = summary["prompt_tokens"]
            values["known_completion_tokens"] = summary["completion_tokens"]
            if summary["token_coverage"] != "complete":
                values["prompt_tokens"] = values["completion_tokens"] = None
            return values

        summary = self.summary(("memory", "agent", "doctor"))
        result = {f"common.{key}": value for key, value in measurements(summary).items()}
        result["common.total_evaluation_cost"] = self.summary()["cost"]
        result["common.total_evaluation_tokens"] = self.summary()["total_tokens"]
        for phase in ("memory", "agent", "judge"):
            roles = ("agent", "doctor") if phase == "agent" else phase
            values = self.summary(roles)
            result.update({f"common.{phase}_{key}": value for key, value in measurements(values).items()})
            result[f"common.{phase}_calls"] = values["tracked_calls"]
            for kind in ("llm", "embedding"):
                values = measurements(self.summary(roles, resource_kind=kind))
                for key in ("prompt_tokens", "completion_tokens", "total_tokens", "token_coverage"):
                    result[f"common.{phase}_{kind}_{key}"] = values[key]
        return result

    def to_dicts(self) -> list[dict]:
        from dataclasses import asdict
        return [asdict(record) for record in self.all()]


class RemoteUsageImporter:
    """Import cumulative remote records once, rejecting counter/history resets."""
    def __init__(self, ledger: UsageLedger):
        self.ledger = ledger
        self._rows: list[dict] = []

    def update(self, rows: list[dict], *, run_id: str | None = None) -> None:
        if len(rows) < len(self._rows) or rows[:len(self._rows)] != self._rows:
            raise ValueError("Remote usage history reset or changed during the run")
        from mematlas.core.provenance import json_sha256
        for index, row in enumerate(rows[len(self._rows):], start=len(self._rows)):
            data = dict(row)
            data.setdefault("record_id", json_sha256({"run_id": run_id, "index": index, "record": row}))
            data.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
            for key, value in current_usage_context().items():
                if key in {"stage", "segment_id", "case_id", "operation", "operation_id"}:
                    data.setdefault(key, value)
            data.setdefault("run_id", run_id)
            self.ledger.add(UsageRecord(**data), imported=True)
        self._rows = [dict(row) for row in rows]


@dataclass
class TraceCollector:
    _traces: list[OperationTrace] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock)
    _journal: Path | None = None
    _identities: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        for trace in self._traces:
            identity, record = self._trace_identity(trace)
            if identity in self._identities and self._identities[identity] != record:
                raise ValueError("Conflicting operation traces have the same operation_id")
            self._identities[identity] = record

    @staticmethod
    def _trace_identity(trace: OperationTrace) -> tuple[str, str]:
        from hashlib import sha256

        record = json.dumps(asdict(trace), ensure_ascii=False, sort_keys=True, default=str)
        operation_id = trace.metadata.get("operation_id")
        identity = ("operation:" + str(operation_id) if operation_id else
                    "record:" + sha256(record.encode("utf-8")).hexdigest())
        return identity, record

    @staticmethod
    def _read_journal(path: Path) -> list[OperationTrace]:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeError as exc:
            raise ValueError(f"Invalid operation trace journal {path}: invalid UTF-8") from exc
        traces = []
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("each record must be an object")
                row["started_at"] = datetime.fromisoformat(row["started_at"])
                trace = OperationTrace(**row)
                if (not isinstance(trace.method_id, str) or not trace.method_id or
                    not isinstance(trace.operation, str) or not trace.operation or
                    not isinstance(trace.success, bool) or
                    not isinstance(trace.duration_ms, (int, float)) or
                    isinstance(trace.duration_ms, bool) or not math.isfinite(trace.duration_ms) or
                    trace.duration_ms < 0 or not isinstance(trace.metadata, dict)):
                    raise ValueError("invalid operation fields")
                if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
                       for value in (trace.input_size, trace.output_size)):
                    raise ValueError("invalid operation sizes")
                if any(value is not None and not isinstance(value, str)
                       for value in (trace.error_type, trace.error_message)):
                    raise ValueError("invalid operation error fields")
                operation_id = trace.metadata.get("operation_id")
                if operation_id is not None and (not isinstance(operation_id, str) or not operation_id):
                    raise ValueError("invalid operation_id")
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Invalid operation trace journal {path}, line {number}: {exc}") from exc
            traces.append(trace)
        return traces

    def bind_journal(self, path: str | Path) -> None:
        self._journal = Path(path).resolve()
        self._journal.parent.mkdir(parents=True, exist_ok=True)

    def add(self, trace: OperationTrace) -> None:
        with self._lock:
            identity, record = self._trace_identity(trace)
            if identity in self._identities:
                if self._identities[identity] != record:
                    raise ValueError("Conflicting operation traces have the same operation_id")
                return
            if self._journal is not None:
                with self._journal.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(trace.to_dict(), ensure_ascii=False, default=str) + "\n")
                    stream.flush()
            self._traces.append(trace)
            self._identities[identity] = record

    def import_journal(self, path: str | Path) -> None:
        """Restore complete traces after resume validation, without counting a record twice.

        Distinct operation IDs retain retries, including failed attempts. Older traces
        without IDs use their complete record as identity. Invalid or conflicting
        records fail before either the collector or its bound journal is changed.
        """
        source = Path(path).resolve()
        imported = self._read_journal(source)
        with self._lock:
            persisted = []
            if self._journal is not None and self._journal.exists():
                persisted = imported if self._journal == source else self._read_journal(self._journal)
            traces: list[OperationTrace] = []
            identities: dict[str, str] = {}
            for trace in [*imported, *persisted, *self._traces]:
                identity, record = self._trace_identity(trace)
                if identity in identities:
                    if identities[identity] != record:
                        raise ValueError("Conflicting operation traces have the same operation_id")
                    continue
                traces.append(trace)
                identities[identity] = record
            if self._journal is not None:
                import tempfile

                fd, name = tempfile.mkstemp(dir=self._journal.parent, prefix=".traces-", suffix=".tmp")
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as stream:
                        for trace in traces:
                            stream.write(json.dumps(trace.to_dict(), ensure_ascii=False, default=str) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(name, self._journal)
                finally:
                    if os.path.exists(name):
                        os.unlink(name)
            self._traces = traces
            self._identities = identities

    def all(self) -> tuple[OperationTrace, ...]:
        with self._lock:
            return tuple(self._traces)

    def clear(self) -> None:
        with self._lock:
            self._traces.clear()
            self._identities.clear()

    def write_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for trace in self.all():
                handle.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")

    def summary(self) -> dict[str, float | int]:
        traces = self.all()
        return {
            "operations": len(traces),
            "failures": sum(not trace.success for trace in traces),
            "duration_ms": sum(trace.duration_ms for trace in traces),
        }
