"""Recovered operation measurements retain attempts and their original attribution."""

import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from mematlas.core.models import OperationTrace
from mematlas.core.telemetry import TraceCollector
from mematlas.evaluation.metrics.common import Evaluator


def trace(operation_id="learn-1", *, success=True, **changes):
    value = OperationTrace(
        "fixture", "add", datetime(2026, 11, 1, tzinfo=timezone.utc), 250, success,
        input_size=3, output_size=2,
        error_type=None if success else "RuntimeError",
        error_message=None if success else "interrupted",
        metadata={
            "operation_id": operation_id, "run_id": "source-run", "segment_id": "segment-1",
            "stage": "training", "case_id": "case-1", "operation": "add",
            "backend_status": {"state": "healthy" if success else "uncertain"},
        },
    )
    return replace(value, **changes)


def write_journal(path, records):
    path.write_text("".join(json.dumps(record.to_dict()) + "\n" for record in records), encoding="utf-8")


def test_recovered_traces_preserve_attribution_and_distinct_retry_attempts(tmp_path):
    source = tmp_path / "source.jsonl"
    failed = trace("attempt-1", success=False)
    succeeded = trace("attempt-2")
    write_journal(source, [failed, succeeded])
    target = tmp_path / "target.jsonl"
    collector = TraceCollector()
    collector.bind_journal(target)
    collector.add(trace("recovery-setup", operation="setup"))

    collector.import_journal(source)
    collector.import_journal(source)

    assert collector.all()[:2] == (failed, succeeded)
    assert collector.summary() == {"operations": 3, "failures": 1, "duration_ms": 750}
    resources = Evaluator._operation_resources(collector)
    assert resources["common.add_count"] == 2
    assert resources["common.add_failures"] == 1
    restored = TraceCollector()
    restored.import_journal(target)
    assert restored.all() == collector.all()
    assert len(target.read_text(encoding="utf-8").splitlines()) == 3


def test_legacy_traces_use_complete_records_including_time_and_outcome(tmp_path):
    source = tmp_path / "legacy.jsonl"
    failed = trace(success=False, metadata={"stage": "training"})
    succeeded = trace(metadata={"stage": "training"})
    later = replace(succeeded, started_at=succeeded.started_at.replace(second=1))
    write_journal(source, [failed, succeeded, succeeded, later])
    collector = TraceCollector()

    collector.import_journal(source)
    collector.import_journal(source)

    assert collector.all() == (failed, succeeded, later)


def test_resume_in_same_directory_does_not_duplicate_history_or_live_records(tmp_path):
    target = tmp_path / "operations.jsonl"
    old = trace("old")
    write_journal(target, [old])
    collector = TraceCollector()
    collector.bind_journal(target)
    live = trace("live")
    collector.add(live)

    collector.import_journal(target)
    collector.import_journal(target)
    collector.add(old)

    assert set(record.metadata["operation_id"] for record in collector.all()) == {"old", "live"}
    assert len(collector.all()) == 2
    assert len(target.read_text(encoding="utf-8").splitlines()) == 2


@pytest.mark.parametrize("invalid", [
    "{unfinished", json.dumps({"started_at": "invalid"}),
    json.dumps({**trace().to_dict(), "duration_ms": -1}),
    json.dumps({**trace().to_dict(), "success": "false"}),
])
def test_corrupt_journal_fails_with_line_and_preserves_existing_records(tmp_path, invalid):
    source = tmp_path / "bad.jsonl"
    source.write_text(json.dumps(trace().to_dict()) + "\n" + invalid, encoding="utf-8")
    target = tmp_path / "current.jsonl"
    collector = TraceCollector()
    collector.bind_journal(target)
    existing = trace("existing")
    collector.add(existing)
    previous = target.read_bytes()

    with pytest.raises(ValueError, match="line 2"):
        collector.import_journal(source)

    assert collector.all() == (existing,)
    assert target.read_bytes() == previous


def test_conflicting_identity_is_rejected_without_partial_import(tmp_path):
    source = tmp_path / "conflict.jsonl"
    existing = trace("existing")
    write_journal(source, [trace("new"), replace(existing, duration_ms=500)])
    collector = TraceCollector()
    collector.add(existing)

    with pytest.raises(ValueError, match="same operation_id"):
        collector.import_journal(source)

    assert collector.all() == (existing,)


def test_clear_removes_identity_cache(tmp_path):
    source = tmp_path / "source.jsonl"
    original = trace()
    write_journal(source, [original])
    collector = TraceCollector()
    collector.import_journal(source)
    collector.clear()
    collector.import_journal(source)
    assert collector.all() == (original,)
