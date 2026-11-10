"""Persistent experiment accounting and checkpoint cost alignment."""
from __future__ import annotations

from time import perf_counter
from typing import Any

from mematlas.core.telemetry import UsageLedger


ACQUISITION_STAGES = ("preparation", "construction", "training", "internal_validation", "recovery", "unclassified")


def cost_summary(ledger: UsageLedger, *, segment_id: str) -> dict[str, Any]:
    stages = sorted({row.stage for row in ledger.all()})
    historical = UsageLedger(**ledger.pricing_spec())
    for record in ledger.all():
        if record.segment_id != segment_id:
            historical.add(record, imported=True)
    return {
        "total": ledger.summary(),
        "current_segment": ledger.summary(segment_id=segment_id),
        "historical": historical.summary(),
        "by_episode": {case: ledger.summary(case_id=case)
                       for case in sorted({row.case_id for row in ledger.all() if row.case_id})},
        "preparation": ledger.summary(stages=("preparation",)),
        "acquisition": ledger.summary(stages=ACQUISITION_STAGES),
        "evaluation": ledger.summary(stages=("evaluation", "measurement")),
        "by_stage": {stage: ledger.summary(stages=(stage,)) for stage in stages},
        "by_component": {role: ledger.summary(role) for role in sorted({row.phase for row in ledger.all()})},
        "by_operation": {operation: summarize_records(ledger, operation=operation)
                         for operation in sorted({row.operation or "unclassified" for row in ledger.all()})},
        "scope": "all_recorded_calls_including_failed_attempts_and_prior_segments",
    }


def summarize_records(ledger, *, operation):
    selected = UsageLedger(**ledger.pricing_spec())
    for record in ledger.all():
        if (record.operation or "unclassified") == operation:
            selected.add(record, imported=True)
    return selected.summary()


def checkpoint_costs(ledger: UsageLedger) -> dict[str, Any]:
    acquisition = ledger.summary(stages=ACQUISITION_STAGES)
    total = ledger.summary()
    return {
        "cumulative_acquisition_cost": acquisition["cost"],
        "cumulative_known_acquisition_cost": acquisition["known_cost"],
        "cumulative_acquisition_tokens": acquisition["total_tokens"],
        "cumulative_experiment_cost": total["cost"],
        "cumulative_experiment_tokens": total["total_tokens"],
        "cost_currency": ledger.currency,
        "cost_coverage": "complete" if acquisition["uncosted_calls"] == 0 else "partial",
    }


def compare_experience_costs(reports, budgets, *, metric="overall_clinical_score"):
    """Compare observed checkpoints at common budgets without interpolating scores."""
    currencies = {report["cost_summary"]["total"]["currency"] for report in reports}
    if len(currencies) != 1:
        raise ValueError("Budget comparisons require one currency")
    panels = {report["run_metadata"]["protocol"]["split_id_sha256"]["primary_test"] for report in reports}
    if len(panels) != 1:
        raise ValueError("Budget comparisons require the same held-out panel")
    rows = []
    for budget in budgets:
        if not isinstance(budget, (int, float)) or isinstance(budget, bool) or budget < 0:
            raise ValueError("Budgets must be nonnegative numeric values")
        for report in reports:
            checkpoints = [point for point in report["batches"] if point.get("split") == "primary_test"
                           and point.get("cost_coverage") == "complete"
                           and isinstance(point.get("cumulative_acquisition_cost"), (int, float))
                           and point["cumulative_acquisition_cost"] <= budget
                           and isinstance(point.get(metric), (int, float))
                           and point.get("evaluation_coverage", 1) == 1
                           and point.get("failed_evaluations", 0) == 0]
            chosen = max(checkpoints, key=lambda point: point["checkpoint"], default=None)
            rows.append({"method_id": report["method_id"], "budget": budget,
                         "currency": next(iter(currencies)), "metric": metric,
                         "checkpoint": chosen["checkpoint"] if chosen else None,
                         "cost": chosen["cumulative_acquisition_cost"] if chosen else None,
                         "score": chosen[metric] if chosen else None,
                         "status": "observed" if chosen else "no_complete_checkpoint_within_budget"})
    return rows


def paired_clinical_comparison(reference, candidate, *, metric="diagnosisAccuracy", bootstrap_samples=2000, seed=0):
    """Patient-paired differences with a patient-cluster bootstrap at one checkpoint."""
    from random import Random
    from statistics import mean
    import math
    def valid_score(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1
    if not isinstance(bootstrap_samples, int) or isinstance(bootstrap_samples, bool) or bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    def values(rows):
        result = {}
        for row in rows:
            patient = str(row["patient_id"])
            report = row.get("report", row)
            if report.get("evaluationStatus", "evaluated") == "evaluated" and valid_score(report.get(metric)):
                result.setdefault(patient, []).append(report[metric])
        return {key: mean(items) for key, items in result.items()}
    left, right = values(reference), values(candidate)
    patients = sorted(left.keys() & right.keys())
    if not patients:
        return {"status": "no_valid_pairs", "paired_patients": 0, "mean_difference": None, "ci95": None}
    differences = [right[patient] - left[patient] for patient in patients]
    rng = Random(seed)
    samples = sorted(mean(rng.choices(differences, k=len(differences))) for _ in range(bootstrap_samples))
    return {"status": "paired_observations", "paired_patients": len(patients),
            "missing_reference_patients": len(right.keys() - left.keys()),
            "missing_candidate_patients": len(left.keys() - right.keys()),
            "mean_difference": mean(differences),
            "ci95": [samples[int(0.025 * (bootstrap_samples - 1))], samples[int(0.975 * (bootstrap_samples - 1))]],
            "uncertainty_scope": "patient_cluster_conditional_on_supplied_runs_and_checkpoint"}


class AccountingWindow:
    def __init__(self, memory):
        self.memory = memory
        self.cursor = memory.services.usage.cursor()
        self.trace_cursor = len(memory.traces.all())
        self.started = perf_counter()

    def finish(self) -> dict[str, Any]:
        ledger = self.memory.services.usage
        usage = ledger.summary(since=self.cursor)
        memory_usage = ledger.summary("memory", since=self.cursor)
        all_traces = self.memory.traces.all()[self.trace_cursor:]
        traces = [row for row in all_traces if not row.operation.startswith("agent_")
                  and row.operation != "judge_scoring"]
        retrievals = [row.duration_ms for row in traces if row.operation == "search" and row.success]
        return {
            "window_usage": usage,
            "combined_cost": usage["cost"],
            "combined_input_tokens": usage["prompt_tokens"] if usage["token_coverage"] == "complete" else None,
            "combined_total_tokens": usage["total_tokens"],
            "known_combined_input_tokens": usage["prompt_tokens"],
            "memory_cost": memory_usage["cost"],
            "memory_prompt_tokens": memory_usage["prompt_tokens"],
            "memory_completion_tokens": memory_usage["completion_tokens"],
            "memory_total_tokens": memory_usage["total_tokens"],
            "memory_tracked_llm_calls": memory_usage["llm_calls"],
            "memory_embedding_calls": memory_usage["embedding_calls"],
            "window_wall_time_ms": (perf_counter() - self.started) * 1000,
            "judging_time_s": sum(row.duration_ms for row in all_traces
                                  if row.operation == "judge_scoring") / 1000,
            "memory_operation_time_ms": sum(row.duration_ms for row in traces),
            "agent_operation_time_ms": sum(row.duration_ms for row in all_traces if row.operation.startswith("agent_")),
            "memory_retrieval_count": len(retrievals),
            "memory_retrieval_time_ms": sum(retrievals),
            "memory_retrieval_mean_ms": sum(retrievals) / len(retrievals) if retrievals else 0.0,
        }
