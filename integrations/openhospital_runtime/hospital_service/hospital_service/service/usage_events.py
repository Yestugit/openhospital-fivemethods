"""Per-attempt model usage, including examinations, repairs and failures."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse
from uuid import uuid4


_ACTIVE = ContextVar("hospital_call_usage", default=None)


class CallUsageStore:
    def __init__(self, journal=None):
        self.instance_id = uuid4().hex
        self._lock = Lock()
        self._rows = []
        self._journal = Path(journal).resolve() if journal else None
        if self._journal:
            self._journal.parent.mkdir(parents=True, exist_ok=True)
            if self._journal.exists():
                data = self._journal.read_bytes()
                lines = data.splitlines(keepends=True)
                for index, line in enumerate(lines):
                    if not line.strip():
                        continue
                    try:
                        self._rows.append(json.loads(line))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        if index != len(lines) - 1 or line.endswith(b"\n"):
                            raise ValueError("Hospital usage journal is corrupt before its final record")
                        from hashlib import sha256
                        record = {"record_id":"incomplete-" + sha256(data).hexdigest(),
                            "timestamp":"recovery", "provider":"journal_recovery", "model":"unknown",
                            "phase":"environment", "prompt_tokens":0, "completion_tokens":0,
                            "usage_known":False, "usage_source":"unknown", "status":"unknown",
                            "stage":"unclassified", "metadata":{"reason":"Interrupted service usage record"}}
                        self._rows.append({"dataset_key":"*", "team_id":"*", "usage":record})
                        # Preserve the damaged original before replacing its unfinished tail.
                        self._journal.with_suffix(".interrupted").write_bytes(data)
                        temporary = self._journal.with_suffix(".recovery.tmp")
                        with temporary.open("w", encoding="utf-8") as stream:
                            stream.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in self._rows))
                            stream.flush()
                            os.fsync(stream.fileno())
                        os.replace(temporary, self._journal)

    @contextmanager
    def scope(self, *, dataset_key, team_id, component, patient_id=None, accounting_context=None):
        context = {key: value for key, value in (accounting_context or {}).items()
                   if key in {"stage", "operation", "run_id", "segment_id", "case_id"}
                   and isinstance(value, str)}
        token = _ACTIVE.set((self, dict(dataset_key=dataset_key, team_id=team_id,
                                        component=component, patient_id=patient_id, accounting=context)))
        try:
            yield
        finally:
            _ACTIVE.reset(token)

    def add(self, row):
        with self._lock:
            if self._journal:
                with self._journal.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            self._rows.append(row)

    def records(self, *, dataset_key, team_id):
        with self._lock:
            return [dict(row["usage"]) for row in self._rows
                    if (row["dataset_key"] == dataset_key and row["team_id"] == team_id)
                    or row["dataset_key"] == row["team_id"] == "*"]


def record_model_call(usage, *, model, endpoint, status="success", attempt=1, **metadata):
    active = _ACTIVE.get()
    if active is None:
        return
    store, context = active
    known = isinstance(usage, dict) and all(isinstance(usage.get(key), int) and not isinstance(usage[key], bool)
                                           and usage[key] >= 0 for key in ("prompt_tokens", "completion_tokens"))
    record = {
        "record_id": uuid4().hex, "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": urlparse(endpoint).netloc, "model": model,
        "phase": context["component"], "case_id": context["patient_id"],
        "prompt_tokens": usage["prompt_tokens"] if known else 0,
        "completion_tokens": usage["completion_tokens"] if known else 0,
        "usage_known": known, "usage_source": "reported" if known else "unknown",
        "status": status, "metadata": {"attempt": attempt, "origin": "hospital_service", **metadata},
        **context["accounting"],
    }
    if isinstance(usage, dict) and isinstance(usage.get("prompt_tokens_details"), dict):
        record["metadata"]["prompt_tokens_details"] = dict(usage["prompt_tokens_details"])
    store.add({**context, "usage": record})
