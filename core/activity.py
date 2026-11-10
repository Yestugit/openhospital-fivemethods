"""Internal durable activity journal. perf_counter uses the host-wide clock.

Separate files per process avoid interprocess append races. Starts are durable
before work; heartbeat bounds an unfinished span after an unclean exit.
"""
from contextlib import contextmanager
from pathlib import Path
import json
import os
import errno
import re
import threading
import time
from uuid import uuid4
from mematlas.core.telemetry import current_usage_context

_lock = threading.Lock()
_PROCESS_INSTANCE = uuid4().hex
_READ_CHUNK_BYTES = 1024 * 1024
_SUMMARY_CACHE = {}
_SUMMARY_LOCK = threading.RLock()


def append(kind, **fields):
    folder = os.environ.get("MEMATLAS_ACTIVITY_DIR")
    if not folder:
        return
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    row = {"kind": kind, "clock_ns": time.perf_counter_ns(), "utc_ns": time.time_ns(),
           "pid": os.getpid(), "process_instance": _PROCESS_INSTANCE, **current_usage_context(), **fields}
    # Background tasks inherit the setup context. Read the current round rather
    # than attributing later batches to the round that created the scheduler.
    try:
        context = json.loads((path / "context.json").read_text(encoding="utf-8"))
        row = {**row, **context, **fields}
    except (OSError, ValueError):
        pass
    with _lock, (path / f"events-{os.getpid()}.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        stream.flush()


@contextmanager
def activity(category, operation, **fields):
    identifier = uuid4().hex
    append("start", span_id=identifier, category=category, operation=operation, **fields)
    try:
        yield identifier
    finally:
        append("end", span_id=identifier)


def _chunk(stream):
    """Bound Windows reads; retry transient descriptor/filter-driver errors."""
    for attempt in range(4):
        try:
            return stream.read(_READ_CHUNK_BYTES)
        except OSError as exc:
            if exc.errno not in {errno.EINVAL, errno.EAGAIN, errno.EINTR} or attempt == 3:
                raise
            time.sleep(.05 * 2**attempt)


def _records(path, *, offset=0, kinds=None):
    """Yield parsed complete records and byte offsets, retaining partial tails.

    kind is the first field in our durable writer. Check it before decoding a
    potentially huge submission payload when the caller only needs timing.
    """
    def parse(raw):
        if kinds is not None:
            match = re.match(rb'\s*\{\s*"kind"\s*:\s*"([^"\\]+)"', raw[:256])
            if match and match[1].decode('ascii') not in kinds:
                return None
        try:
            value = json.loads(raw)
            return value if kinds is None or value.get("kind") in kinds else None
        except (ValueError, UnicodeDecodeError):
            return None

    with Path(path).open("rb") as stream:
        stream.seek(offset)
        pending = b""
        while True:
            chunk = _chunk(stream)
            if not chunk:
                break
            pending += chunk
            lines = pending.split(b"\n")
            pending = lines.pop()
            for line in lines:
                offset += len(line) + 1
                yield parse(line), offset
        # Tests and some imported journals have a complete final JSON record
        # without a newline. An incomplete writer tail is retried next poll.
        if pending:
            try:
                json.loads(pending)
            except (ValueError, UnicodeDecodeError):
                return
            yield parse(pending), offset + len(pending)


def iter_rows(folder, *, kinds=None):
    for path in Path(folder).glob("events-*.jsonl"):
        for record, _ in _records(path, kinds=kinds):
            if record is not None:
                yield record


def rows(folder, *, kinds=None):
    return list(iter_rows(folder, kinds=kinds))


def _timing_rows(folder):
    """Cache only compact timing events and scan appended bytes on each poll."""
    kinds = {"start", "end", "heartbeat"}
    key = str(Path(folder).resolve())
    with _SUMMARY_LOCK:
        files = _SUMMARY_CACHE.setdefault(key, {})
        paths = list(Path(folder).glob("events-*.jsonl"))
        existing = {str(p) for p in paths}
        for removed in set(files)-existing:
            del files[removed]
        for path in paths:
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            state = files.get(str(path))
            if state is None or state["identity"] != identity or stat.st_size < state["size"] or (
                    stat.st_size == state["size"] and stat.st_mtime_ns != state["mtime"]):
                state = {"identity":identity, "offset":0, "events":[], "size":0, "mtime":0}
                files[str(path)] = state
            # Advancing one record at a time also makes a read retry safe after
            # partial progress. No incomplete tail is cached as a full event.
            for record, offset in _records(path, offset=state["offset"], kinds=kinds):
                if record is not None:
                    state["events"].append(record)
                state["offset"] = offset
            state.update(size=stat.st_size, mtime=stat.st_mtime_ns)
        return [event for state in files.values() for event in state["events"]]


def union_seconds(intervals):
    end = None
    total = 0
    for start, stop in sorted(intervals):
        if stop < start:
            raise ValueError("Clock interval reversed")
        total += max(0, stop - max(start, end if end is not None else start))
        end = max(end or stop, stop)
    return total / 1e9


def summarize(folder, round_index=None):
    events = _timing_rows(folder)
    starts = {r["span_id"]: r for r in events if r["kind"] == "start"}
    ends = {r["span_id"]: r for r in events if r["kind"] == "end"}
    heartbeats = {}
    for r in events:
        if r["kind"] == "heartbeat":
            identity = (r["pid"],r.get("process_instance"))
            heartbeats[identity] = max(heartbeats.get(identity, 0), r["clock_ns"])
    groups, incomplete, uncertainty = {}, [], 0
    for key, start in starts.items():
        if round_index is not None and start.get("round") != round_index:
            continue
        if key in ends:
            stop = ends[key]["clock_ns"]
        else:
            stop = max(start["clock_ns"], heartbeats.get((start["pid"],start.get("process_instance")), start["clock_ns"]))
            incomplete.append(key)
            uncertainty += 5.0
        groups.setdefault(start["category"], []).append((start["clock_ns"], stop))
    mapping = {"round": "round_total_s", "foreground": "memory_foreground_wait_s",
               "background": "memory_background_active_s", "persistence": "memory_persistence_s",
               "measurement": "memory_measurement_s", "recovery_wait": "automatic_recovery_wait_s"}
    result = {name: union_seconds(groups.get(category, [])) for category, name in mapping.items()}
    result["memory_method_wall_s"] = union_seconds(groups.get("foreground", []) + groups.get("background", []))
    result.update(timing_complete=not incomplete, unfinished_spans=incomplete,
                  heartbeat_interval_s=5, unobserved_tail_bound_s=None if incomplete else 0)
    return result
