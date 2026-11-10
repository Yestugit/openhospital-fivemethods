#!/usr/bin/env python3
"""Summarize existing CLI evidence without loading plugins or making API calls.

Usage: python test/cli_e2e/summarize.py REPORT_DIR [REPORT_DIR ...]

Only the supplied directories' results.json files are counted; immutable copies
under experiments/ are not scanned. SQLite connections use mode=ro and
query_only. Output contains counts, identifiers and hashes, never task content,
provider configuration, raw exceptions, credentials or complete native state.
An absent report is explicitly incomplete, even if a CLI log already exists.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3


def mapping(value):
    return value if isinstance(value, dict) else {}


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    rows, invalid = [], 0
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("Expected an object")
                    rows.append(row)
                except (ValueError, TypeError):
                    invalid += 1
    return rows, invalid


def key_value(value):
    if isinstance(value, list):
        return tuple(key_value(item) for item in value)
    if isinstance(value, dict):
        return tuple((key_value(key), key_value(item)) for key, item in value.items())
    return value


def decode_state(value):
    """Read typed snapshot containers without importing native code or NumPy."""
    if isinstance(value, list):
        return [decode_state(item) for item in value]
    if not isinstance(value, dict):
        return value
    kind = value.get("kind")
    if kind in {"mapping", "ordered_mapping"}:
        return {key_value(decode_state(key)): decode_state(item)
                for key, item in value["items"]}
    if kind in {"list", "tuple", "set", "frozenset", "ndarray"}:
        return decode_state(value["items"])
    if kind == "task_features":
        return decode_state(value["fields"])
    if kind in {"date", "datetime"}:
        return value["value"]
    return {key: decode_state(item) for key, item in value.items()}


def dictionaries(value):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from dictionaries(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from dictionaries(item)


def scalar_values(value, field):
    return [item[field] for item in dictionaries(value) if number(item.get(field))]


def numeric_leaves(value):
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in numeric_leaves(item)]
    if isinstance(value, (list, tuple)):
        return [leaf for item in value for leaf in numeric_leaves(item)]
    return [value] if number(value) else []


def distribution(values):
    values = [value for value in values if number(value)]
    counts = Counter(format(value, ".12g") for value in values)
    return {"count": len(values), "min": min(values) if values else None,
            "max": max(values) if values else None, "sum": sum(values),
            "value_counts": dict(sorted(counts.items()))}


def q_stats(payloads):
    # MemRL stores Q fields below payload/metadata; do not assume a flat row.
    return {field: distribution(scalar_values(payloads, field))
            for field in ("q_value", "q_visits", "retrieval_count")}


def grouped_size(value):
    if isinstance(value, dict):
        return sum(len(item) if isinstance(item, (list, tuple, dict)) else 1
                   for item in value.values())
    return len(value) if isinstance(value, (list, tuple)) else 0


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def resolve_path(value, report_dir):
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, report_dir / "workdir" / path,
                  report_dir / path, *(parent / path for parent in report_dir.parents)]
    return next((candidate.resolve() for candidate in candidates if candidate.exists()),
                candidates[0].resolve())


def artifact_path(result, report_dir):
    meta = mapping(result.get("run_metadata"))
    config = mapping(meta.get("config"))
    raw = mapping(config.get("runtime")).get("artifact_dir")
    if raw:
        return resolve_path(raw, report_dir)
    run_id = meta.get("run_id")
    return report_dir / "artifacts" / "runs" / str(run_id) if run_id else report_dir / "artifacts"


def storage_path(result, artifact, report_dir):
    config = mapping(mapping(result.get("run_metadata")).get("config"))
    params = mapping(mapping(config.get("method")).get("params"))
    # MemRL always persists its sidecar under the allocated artifact directory.
    method = result.get("method_id")
    if method != "memrl" and params.get("storage_path"):
        return resolve_path(params["storage_path"], report_dir)
    return artifact / str(method)


def operation_summary(traces):
    operations, failed, stages = Counter(), Counter(), Counter()
    for trace in traces:
        operation = str(trace.get("operation", "unknown"))
        stage = str(mapping(trace.get("metadata")).get("stage", "unknown"))
        operations[operation] += 1
        stages[(stage, operation)] += 1
        if trace.get("success") is False:
            failed[operation] += 1
    return {"total": len(traces), "by_operation": dict(sorted(operations.items())),
            "failed": sum(failed.values()), "failed_by_operation": dict(sorted(failed.items())),
            "by_stage": [{"stage": stage, "operation": operation, "count": count}
                         for (stage, operation), count in sorted(stages.items())]}


def usage_summary(path, run_id):
    rows, invalid = read_jsonl(path)
    seen, groups = set(), Counter()
    duplicate_records = excluded = without_id = 0
    for row in rows:
        if row.get("run_id") and run_id and row["run_id"] != run_id:
            excluded += 1
            continue
        record_id = row.get("record_id")
        if record_id and record_id in seen:
            duplicate_records += 1
            continue
        if record_id:
            seen.add(record_id)
        else:
            without_id += 1
        key = tuple(str(row.get(field) or "unknown")
                    for field in ("phase", "resource_kind", "stage", "status"))
        groups[key] += 1
    return {"available": path.is_file(), "records": sum(groups.values()),
            "invalid_jsonl_rows": invalid, "duplicate_record_ids": duplicate_records,
            "records_without_id": without_id, "other_run_records_excluded": excluded,
            "groups": [{"phase": phase, "resource_kind": kind, "stage": stage,
                        "status": status, "count": count}
                       for (phase, kind, stage, status), count in sorted(groups.items())],
            "coverage": "Framework ledger only; direct native provider calls may bypass it."}


def log_summary(path):
    result = {"available": path.is_file(), "level_counts": {}, "origins": {}, "signals": {}}
    if not path.is_file():
        return result
    levels, origins, signals = Counter(), Counter(), Counter()
    ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    level_pattern = re.compile(r"(?:^|[\s|])(CRITICAL|ERROR|WARNING|WARN|INFO|DEBUG)(?=$|[\s|:])")
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = ansi.sub("", line)
            lowered = line.lower()
            match = level_pattern.search(line)
            if match:
                level = "WARNING" if match.group(1) == "WARN" else match.group(1)
                levels[level] += 1
                if "ael.native.calls" in lowered or "[ael-native-call]" in line:
                    origin = "adapter_native_call_trace"
                elif "-adapter]" in line or "mematlas.methods" in line:
                    origin = "adapter"
                elif any(marker in lowered for marker in ("litellm", "httpx", "httpcore", "openai.")):
                    origin = "provider_library"
                elif any(marker in lowered for marker in
                         ("memos.", "memrl.", "memq.", "ael.", "hmf.", "mas.", "chromadb.")):
                    origin = "upstream_native"
                else:
                    origin = "framework_or_unclassified"
                origins[(origin, level)] += 1
            for name, marker in {
                "empty_update_request": "empty update request",
                "traceback": "traceback (most recent call last)",
                "capability_mismatch": "capabilitymismatcherror",
                "plan_access_rejected": "cli e2e forbids runtime access to plan/",
                "missing_api_key": "api_key client option must be set",
            }.items():
                if marker in lowered:
                    signals[name] += 1
    result.update(level_counts=dict(sorted(levels.items())),
                  origins=[{"origin": origin, "level": level, "count": count}
                           for (origin, level), count in sorted(origins.items())],
                  signals=dict(sorted(signals.items())))
    result["coverage"] = "Recognized log lines only; counts are diagnostic, not proof of error absence."
    return result


def memrl_summary(state):
    scopes, errors = [], []
    for key, service in mapping(state.get("services")).items():
        dump = Path(service["snapshot_root"]) / "cube" / "textual_memory.json"
        scope = {"scope_id": str(key), "native_dump_available": dump.is_file()}
        if dump.is_file():
            try:
                payloads = read_json(dump)
                scope.update(native_records=len(payloads) if isinstance(payloads, list) else None,
                             native_dump_sha256=hashlib.sha256(dump.read_bytes()).hexdigest(),
                             **q_stats(payloads))
            except (OSError, ValueError, TypeError) as error:
                errors.append({"scope_id": str(key), "error_type": type(error).__name__})
        scopes.append(scope)
    return {"scopes": scopes, "native_scopes": len(scopes),
            "pending_batch": grouped_size(state.get("pending_batch")),
            "pending_queries": grouped_size(state.get("pending_queries")),
            "batch_open": state.get("batch_open"), "read_errors": errors}


def memq_summary(state):
    adapter = mapping(decode_state(state.get("adapter")))
    scopes = []
    for key, scope in mapping(state.get("scopes")).items():
        native = mapping(scope.get("native"))
        fields = mapping(decode_state(native.get("fields")))
        store_fields = mapping(decode_state(native.get("store_fields")))
        payloads = store_fields.get("_payloads")
        if payloads is None and mapping(native.get("store")).get("payloads.json"):
            saved = json.loads(base64.b64decode(native["store"]["payloads.json"], validate=True))
            payloads = mapping(saved).get("payloads")
        scopes.append({"scope_id": str(key),
                       "native_records": len(payloads) if isinstance(payloads, (list, dict)) else None,
                       "q_max": fields.get("_q_max"),
                       "eligibility_trace_entries": len(mapping(fields.get("_eligibility_traces"))),
                       **q_stats(payloads)})
    return {"scopes": scopes, "native_scopes": len(scopes),
            **{name: grouped_size(adapter.get(name)) for name in
               ("pending_batch", "pending_edges", "deferred_values", "last_retrievals")},
            **{name: adapter.get(name) for name in
               ("batch_open", "epoch_open", "current_epoch", "total_epochs")}}


def ael_summary(state):
    adapter = mapping(decode_state(state.get("adapter")))
    histories = mapping(adapter.get("trajectory_history"))
    days = mapping(adapter.get("day_indices"))
    scopes = []
    for key, scope in mapping(state.get("scopes")).items():
        native = mapping(scope.get("native"))
        memory = mapping(decode_state(native.get("memory")))
        controller = mapping(decode_state(native.get("controller")))
        fields = mapping(decode_state(native.get("controller_fields")))
        entries = list(mapping(memory.get("entries")).values())
        arms = mapping(controller.get("memory_bandit"))
        scopes.append({
            "scope_id": str(key), "native_records": len(entries),
            "episode_counter": memory.get("episode_counter"),
            "tiers": dict(sorted(Counter(str(entry.get("tier")) for entry in entries).items())),
            "retrieval_count": distribution(scalar_values(entries, "retrieval_count")),
            "usefulness_sum": distribution(scalar_values(entries, "usefulness_sum")),
            "controller_episodes_elapsed": fields.get("_episodes_elapsed", controller.get("episodes_elapsed")),
            "trajectory_history_length": len(histories.get(key, [])),
            "day_index": days.get(key),
            "memory_bandit": [{"arm": str(arm), **{name: value.get(name)
                                for name in ("alpha", "beta", "pull_count")}}
                              for arm, value in arms.items() if isinstance(value, dict)],
        })
    return {"scopes": scopes, "native_scopes": len(scopes),
            "trajectory_history_lengths": [{"scope_id": str(key), "length": len(history)}
                                           for key, history in histories.items()],
            "day_indices": [{"scope_id": str(key), "day_index": day} for key, day in days.items()],
            "feedback_count": None,
            "feedback_count_limit": "Native state has no total/negative mark_useful call counter; usefulness_sum is not total feedback."}


def policy_summary(policy):
    policy = mapping(decode_state(policy))
    stats = mapping(policy.get("stats"))
    return {"q_values": distribution(numeric_leaves(policy.get("q"))),
            "action_counts": distribution(numeric_leaves(policy.get("counts"))),
            "total_selections": policy.get("total_selections", policy.get("total")),
            "stats": {name: stats[name] for name in
                      ("updates", "total_reward", "explorations", "exploitations")
                      if number(stats.get(name))}}


def chroma_summary(path):
    result = {"path": str(path), "access": "sqlite mode=ro; PRAGMA query_only=ON"}
    connection = None
    try:
        connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        connection.execute("PRAGMA query_only=ON")
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table in ("embeddings", "collections"):
            if table in tables:
                result[table] = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if "embedding_metadata" in tables:
            rows = connection.execute(
                "SELECT string_value,COUNT(*) FROM embedding_metadata "
                "WHERE key='chroma:document' GROUP BY string_value").fetchall()
            result["document_groups"] = [{"sha256": digest(str(document)), "rows": count}
                                         for document, count in rows]
            result["document_multiplicity"] = dict(sorted(Counter(str(count) for _, count in rows).items()))
    except (sqlite3.Error, OSError) as error:
        result["error_type"] = type(error).__name__
    finally:
        if connection is not None:
            connection.close()
    return result


def memcon_summary(state, root, traces):
    scopes = []
    for key, scope in mapping(state.get("instances")).items():
        native = mapping(scope.get("native"))
        collection = mapping(native.get("collection"))
        graph = mapping(native.get("task_graph"))
        scopes.append({"scope_id": str(key), "native_records": len(collection.get("records", [])),
                       "task_graph_nodes": len(graph.get("nodes", [])),
                       "task_graph_edges": len(graph.get("edges", [])),
                       "policy": policy_summary(native.get("policy"))})
    databases = [chroma_summary(path) for path in sorted(root.rglob("chroma.sqlite3"))] if root.is_dir() else []
    policies = []
    for path in sorted(root.rglob("policy_q.json")) if root.is_dir() else []:
        try:
            policies.append({"path": str(path), **policy_summary(read_json(path))})
        except (OSError, ValueError, TypeError) as error:
            policies.append({"path": str(path), "error_type": type(error).__name__})
    learns = sum(trace.get("operation") == "learn" and trace.get("success") is True
                 and mapping(trace.get("metadata")).get("stage") == "training" for trace in traces)
    known_rows = [db["embeddings"] for db in databases if "embeddings" in db]
    rows = sum(known_rows) if known_rows else None
    return {"scopes": scopes, "native_scopes": len(scopes) if state else None,
            "chroma_databases": databases, "disk_policy_files": policies,
            "persisted_native_records": rows, "successful_training_learns": learns,
            "records_per_training_learn": rows / learns if rows is not None and learns else None,
            "double_write_evidence": "A ratio of two is consistent with native double writes; counts alone do not prove call order.",
            "policy_persistence_limit": "Native policy auto-saves every five updates; an absent/stale file does not prove no learning."}


def native_summary(result, artifact, report_dir, traces):
    root = storage_path(result, artifact, report_dir)
    path = root / "adapter_state.json"
    summary = {"storage_root": str(root), "adapter_state_available": path.is_file()}
    state = {}
    if path.is_file():
        try:
            state = mapping(read_json(path))
            summary["schema_version"] = state.get("schema_version")
            summary["profile"] = state.get("profile")
            summary["adapter_state_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        except (OSError, ValueError, TypeError) as error:
            summary["error_type"] = type(error).__name__
    method = result.get("method_id")
    try:
        if method == "memcon":
            summary.update(memcon_summary(state, root, traces))
        elif state and method in {"memrl", "memq", "ael"}:
            summary.update({"memrl": memrl_summary, "memq": memq_summary, "ael": ael_summary}[method](state))
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        summary["extraction_error_type"] = type(error).__name__
    return summary


def prediction_summary(predictions):
    items = []
    for prediction in predictions:
        confirmed = prediction.get("retrieved") or []
        contribution = mapping(mapping(prediction.get("metadata")).get("contribution"))
        candidates = contribution.get("items") or []
        metadata = mapping(contribution.get("metadata"))
        item = {"case_id": prediction.get("case_id"), "confirmed_hits": len(confirmed),
                      "raw_candidates": len(candidates),
                      "confirmed_memory_ids": [item.get("memory_id") for item in confirmed if isinstance(item, dict)],
                      "candidate_memory_ids": [item.get("memory_id") for item in candidates if isinstance(item, dict)]}
        if metadata.get("method_id") == "ael":
            selection = mapping(metadata.get("controller_selection"))
            item["ael_selection"] = {
                **{name: metadata[name] for name in
                   ("memory_policy", "native_hit_count", "backend", "runner_mode", "profile", "_warmup")
                   if name in metadata},
                "controller_selection": {name: selection[name]
                                         for name in ("planner", "tool_config", "memory_policy")
                                         if name in selection},
                "source": "prediction.metadata.contribution.metadata",
            }
        items.append(item)
    return {"count": len(items), "confirmed_hits": sum(item["confirmed_hits"] for item in items),
            "raw_candidates": sum(item["raw_candidates"] for item in items), "cases": items}


def summarize_result(result, report_dir):
    metadata = mapping(result.get("run_metadata"))
    run_id = metadata.get("run_id")
    artifact = artifact_path(result, report_dir)
    traces = result.get("traces") or []
    trace_source, invalid = "results.json", 0
    if not traces:
        traces, invalid = read_jsonl(artifact / "operations.jsonl")
        trace_source = "operations.jsonl" if traces else "unavailable"
    operations = operation_summary(traces)
    operations.update(source=trace_source, invalid_jsonl_rows=invalid)
    resource = mapping(result.get("resource_metrics"))
    selected = {key: value for key, value in resource.items()
                if any(marker in key for marker in ("failed_calls", "tokens", "wall_time"))
                and (value is None or number(value))}
    failures = result.get("failures") or []
    summary = {
        "method_id": result.get("method_id"), "run_id": run_id,
        "framework_status": result.get("status"), "protocol_id": metadata.get("protocol_id"),
        "benchmark_id": result.get("benchmark_id"), "artifact_dir": str(artifact),
        "framework_failures": len(failures),
        "failure_types": [{key: failure.get(key) for key in ("stage", "error_type")}
                          for failure in failures if isinstance(failure, dict)],
        "resource_metrics": selected, "operations": operations,
        "usage": usage_summary(artifact / "usage.jsonl", run_id),
        "predictions": prediction_summary(result.get("predictions") or []),
        "native": native_summary(result, artifact, report_dir, traces),
        "frozen_state_evidence": {
            "independently_measured_before_after_hash": False,
            "status": "not_independently_measured",
            "successful_save_state_operations": sum(t.get("operation") == "save_state" and t.get("success") is True for t in traces),
            "successful_load_state_operations": sum(t.get("operation") == "load_state" and t.get("success") is True for t in traces),
            "limit": "Existing traces/instrumentation contain no before/after state hashes; final file hashes and successful save/load calls do not prove frozen equality.",
        },
    }
    protocol = str(metadata.get("protocol_id", ""))
    training_retrieves = sum(t.get("operation") == "retrieve"
                             and mapping(t.get("metadata")).get("stage") == "training" for t in traces)
    summary["learning_evidence"] = {
        "training_retrieve_operations": training_retrieves,
        "confirmed_retrieval_credit_verified": False,
        "limit": ("Experience training ingests history without retrieve; held-out reads do not learn outcomes. "
                  "This CLI run cannot prove retrieved-memory Q/TD credit updates."
                  if "experience" in protocol else
                  "Operation counts and final state alone do not establish algorithm update formulas or feedback credit."),
    }
    return summary


def summarize_report(path):
    report_dir = (path.parent if path.name == "results.json" else path).resolve()
    results_file = report_dir / "results.json"
    summary = {"report_dir": str(report_dir), "results_available": results_file.is_file(),
               "cli_log": log_summary(report_dir / "cli.log"), "results": []}
    if not results_file.is_file():
        summary.update(evidence_status="incomplete", reason="No results.json; CLI completion has not been established.")
        return summary
    try:
        results = read_json(results_file)
        if not isinstance(results, list) or not results:
            summary.update(evidence_status="incomplete", reason="results.json contains no evaluation results.")
            return summary
        for result in results:
            if not isinstance(result, dict):
                raise TypeError("Evaluation result must be an object")
            summary["results"].append(summarize_result(result, report_dir))
        summary["evidence_status"] = "reported"
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        summary.update(evidence_status="unreadable", error_type=type(error).__name__)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dirs", metavar="REPORT_DIR", nargs="+", type=Path)
    parser.add_argument("--compact", action="store_true", help="Emit JSON on one line")
    args = parser.parse_args()
    reports = [summarize_report(path) for path in args.report_dirs]
    counts = Counter(report["evidence_status"] for report in reports)
    payload = {"schema_version": 1, "read_only": True,
               "summary": {"requested_reports": len(reports), **dict(sorted(counts.items()))},
               "reports": reports}
    print(json.dumps(payload, ensure_ascii=False, indent=None if args.compact else 2, allow_nan=False))


if __name__ == "__main__":
    main()
