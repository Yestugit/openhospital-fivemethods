"""Experiment-only round instrumentation; no additions to the Memory API."""
import json
import os
from pathlib import Path
from mematlas.core.activity import activity, summarize, append, rows, iter_rows
from mematlas.core.files import atomic_write_json
from mematlas.core.models import content_text
from .inspection import count


def request_inventory(method_dir):
    """Rebuild round costs from all durable attempts, deduplicating record IDs.

    A resumed batch's callback window contains only its remaining work. It must
    not stand in for the actual whole-round request count or token cost.
    """
    from datetime import datetime
    import re
    from mematlas.core.telemetry import UsageLedger
    root = Path(method_dir)
    ledger = UsageLedger()
    for pattern in ("attempt-*/evaluation/runs/*/usage.jsonl", "attempt-*/memory/runs/*/usage.jsonl"):
        for journal in sorted(root.glob(pattern)):
            ledger.import_journal(journal)
    events = rows(root / "activity", kinds={"start", "end"})
    starts = [r for r in events if r["kind"] == "start"]
    ends = {r["span_id"]:r for r in events if r["kind"] == "end"}
    operations = {r["operation_id"]:r.get("round") for r in starts if r.get("operation_id")}
    intervals = [(r["utc_ns"], ends[r["span_id"]]["utc_ns"], r.get("round")) for r in starts
        if r["span_id"] in ends and r.get("round") is not None]
    assigned, unassigned = {}, []
    for record in ledger.to_dicts():
        match = re.search(r"batch:(\d+):patient:", record.get("case_id") or "")
        index = int(match[1]) if match else operations.get(record.get("operation_id"))
        if index is None:
            try:
                timestamp = int(datetime.fromisoformat(record["timestamp"]).timestamp()*1e9)
                candidates = [(end-start, round_) for start,end,round_ in intervals if start <= timestamp <= end]
                if candidates:
                    index = min(candidates)[1]
            except (ValueError, TypeError):
                pass
        record["attributed_round"] = index
        (assigned.setdefault(index, []) if index is not None else unassigned).append(record)
    return assigned, unassigned


def request_summary(records):
    from mematlas.core.telemetry import UsageLedger, UsageRecord
    ledger = UsageLedger()
    for row in records:
        ledger.add(UsageRecord(**{k:v for k,v in row.items() if k != "attributed_round"}), imported=True)
    result = {"actual_request_count":len(records), "actual_failed_request_count":sum(r["status"]!="success" for r in records),
        "actual_unknown_usage_requests":sum(not r["usage_known"] for r in records)}
    for label,phase,kind in (("doctor","doctor","llm"),("memory_llm","memory","llm"),
            ("embedding",None,"embedding"),("hospital",("patient","examination","judge"),"llm")):
        total=ledger.summary(phase,resource_kind=kind)
        result.update({f"actual_{label}_calls":total["tracked_calls"],
            f"actual_{label}_known_tokens":total["known_total_tokens"],
            f"actual_{label}_tokens":total["total_tokens"],
            f"actual_{label}_failed_calls":total["failed_calls"]})
    return result


def input_counts(memory, tokenizer_dir, journal, *, cache=None):
    # Immutable accepted identities let case-level sampling tokenize new data
    # once, rather than repeatedly tokenizing all historical payloads.
    cache = cache if cache is not None else {}
    units_cache = cache.setdefault("units", {})
    events_cache = cache.setdefault("events", {})
    transfer_cache = cache.setdefault("submissions", {})
    history = memory._history
    seen, events, payload = set(), [], 0
    for identity in sorted(history.processed_units):
        unit = history.unit(identity)
        if identity not in units_cache:
            units_cache[identity] = count(json.dumps(unit.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")), tokenizer_dir)
        payload += units_cache[identity]
        for event in unit.events:
            key = history.event_identity(event)
            if key in seen:
                continue
            seen.add(key)
            if key not in events_cache:
                events_cache[key] = count(content_text(event.content), tokenizer_dir)
            events.append({"event_identity": key, "tokens": events_cache[key]})
    transfer = 0
    for row in iter_rows(journal, kinds={"submission"}):
        key = (row.get("pid"), row.get("process_instance"), row.get("clock_ns"))
        if key not in transfer_cache:
            transfer_cache[key] = count(json.dumps(row["unit"], ensure_ascii=False, sort_keys=True, separators=(",", ":")), tokenizer_dir)
        transfer += transfer_cache[key]
    return {"rsi_tokens": sum(r["tokens"] for r in events), "payload_tokens": payload,
            "submitted_payload_tokens": transfer, "additional_attempt_payload_tokens": max(0, transfer - payload),
            "accepted_event_ledger": events, "accepted_units": len(history.processed_units)}


class RoundMeter:
    def __init__(self, engine):
        self.engine = engine
        self.folder = Path(os.environ["MEMATLAS_ACTIVITY_DIR"])
        self.options = json.loads(os.environ["MEMATLAS_EFFICIENCY_OPTIONS"])
        self.active = None
        self.round = 0
        self.count_cache = {}

    async def inventory(self):
        memory = self.engine.memory.core_memory
        await memory._freeze_backend()
        try:
            # A batch already in flight may finish while freeze waits. It is
            # algorithm work, not a model invocation caused by this inspector.
            before = memory.services.usage.cursor()
            method = memory._method
            if hasattr(method, "_inspect"):
                inventory = await method._inspect(self.options)
            else:
                from .inspection import inspect_native
                inventory = await inspect_native(method, self.options)
            counts = input_counts(memory, self.options["tokenizer_dir"], self.folder, cache=self.count_cache)
            if memory.services.usage.cursor() != before:
                raise RuntimeError("Measurement called a model")
            return {**inventory, **counts}
        finally:
            await memory._thaw_backend()

    def case_path(self, position):
        return self.folder / "cases" / f"case-{position:04d}.json"

    def write_case(self, position, result, *, phase):
        from datetime import datetime, timezone
        from mematlas.core.provenance import json_sha256
        from mematlas.evaluation.artifacts import recovery_checkpoint
        checkpoint = None
        if position:
            observed, path, payload = recovery_checkpoint(self.engine.output_dir)
            if observed != position:
                raise ValueError("Case measurement does not match committed recovery position")
            checkpoint = {"position": observed, "payload_sha256": payload["payload_sha256"], "path": str(path)}
        baseline_file = self.folder / "measurement-0.json"
        baseline = json.loads(baseline_file.read_text("utf-8")) if baseline_file.exists() else result
        # Inventories contain identities and counts, never duplicated memory text.
        body = {"schema":1, "method":self.options["method"], "case_position":position,
            "round":self.round if position else 0, "phase":phase,
            "sampled_at":datetime.now(timezone.utc).isoformat(), "checkpoint":checkpoint,
            **{k:v for k,v in result.items() if k not in {"operations","accepted_event_ledger","unfinished_spans"}},
            "storage_growth_bytes":result["storage_bytes"]-baseline["storage_bytes"]}
        # A case sample's round is its actual round, not a reused baseline value.
        body["round"] = self.round if position else 0
        body["payload_sha256"] = json_sha256(body)
        atomic_write_json(self.case_path(position), body)

    async def case(self, position):
        # Sampling pauses the round clock; inspection costs stay separately
        # measurable and do not masquerade as native memory algorithm time.
        self.close()
        try:
            with activity("measurement", "case_native_inventory", case_position=position):
                result = await self.inventory()
                self.write_case(position, result, phase="case_accepted")
        finally:
            self.active = activity("round", "round_active")
            self.active.__enter__()

    async def measure(self, index):
        with activity("measurement", "native_inventory"):
            data = await self.inventory()
        result = {"round": index, **data, **summarize(self.folder, index)}
        zero = self.folder / "measurement-0.json"
        baseline = json.loads(zero.read_text(encoding="utf-8"))["storage_bytes"] if zero.exists() else result["storage_bytes"]
        result["storage_growth_bytes"] = result["storage_bytes"] - baseline
        previous = self.folder / f"measurement-{max(0,index-1)}.json"
        prior = json.loads(previous.read_text(encoding="utf-8")) if previous.exists() else {}
        result["round_rsi_tokens"] = data["rsi_tokens"] - prior.get("rsi_tokens",0)
        result["memory_s_per_1k_rsi_tokens"] = result["memory_method_wall_s"] * 1000 / result["round_rsi_tokens"] if result["round_rsi_tokens"] else None
        events = rows(self.folder)
        ends = {r["span_id"]:r["clock_ns"] for r in events if r["kind"] == "end"}
        operations = {}
        for row in events:
            if row["kind"] != "start" or row.get("round") != index:
                continue
            operation = row["operation"]
            stats = operations.setdefault(operation, {"calls":0,"completed_calls":0,"latencies_s":[]})
            stats["calls"] += 1
            if row["span_id"] in ends:
                stats["completed_calls"] += 1
                stats["latencies_s"].append((ends[row["span_id"]]-row["clock_ns"])/1e9)
        from statistics import median
        for stats in operations.values():
            values = sorted(stats["latencies_s"])
            stats["median_s"] = median(values) if values else None
            stats["p95_s"] = values[max(0,__import__("math").ceil(len(values)*.95)-1)] if values else None
        result["operations"] = operations
        atomic_write_json(self.folder / f"measurement-{index}.json", result)
        return result

    async def start(self, index):
        self.round = index
        atomic_write_json(self.folder / "context.json", {"method": self.options["method"], "round": index,
            "segment_id": self.engine._segment_id, "attempt": int(os.environ.get("MEMATLAS_ATTEMPT", "1"))})
        if not (self.folder / "measurement-0.json").exists():
            baseline = await self.measure(0)
            self.write_case(0, baseline, phase="empty_baseline")
        # A crash after checkpoint commit but before sampling must fill the
        # missing point from restored state, without replaying the clinical case.
        position = self.engine._position
        if position and not self.case_path(position).exists():
            await self.case(position)
            self.close()
        self.active = activity("round", "round_active")
        self.active.__enter__()

    async def end(self, index):
        self.close()
        result = await self.measure(index)
        self.write_case(self.engine._position, result, phase="round_drained")
        self.engine._curve[-1]["efficiency"] = {k: v for k, v in result.items()
            if k not in {"storage_files", "text_objects", "accepted_event_ledger"}}
        self.active = activity("round", "round_commit")
        self.active.__enter__()

    def seal(self, index):
        self.close()
        path = self.folder / f"measurement-{index}.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        result.update(summarize(self.folder, index))
        atomic_write_json(path, result)
        self.engine._curve[-1]["efficiency"].update(summarize(self.folder, index))

    def close(self):
        if self.active is not None:
            self.active.__exit__(None, None, None)
            self.active = None


def case_measurements(method_dir, committed_position):
    """Return one verified point per committed case; retain flat size segments."""
    from mematlas.core.provenance import json_sha256
    points = []
    for path in sorted((Path(method_dir) / "activity/cases").glob("case-*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        checked = dict(value)
        digest = checked.pop("payload_sha256", None)
        if value.get("schema") != 1 or digest != json_sha256(checked):
            raise ValueError("Case measurement checksum mismatch: " + str(path))
        if value["case_position"] <= committed_position:
            points.append(value)
    return points
