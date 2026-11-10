from __future__ import annotations
from statistics import mean
from typing import Any, Iterable, Mapping
from .feedback import valid_score


def aggregate_openhospital_evaluations(records: Iterable[Mapping[str, Any]], *, expected_count: int | None = None) -> dict[str, Any]:
    fields = {
        "diagnosis_accuracy": "diagnosisAccuracy",
        "examination_precision": "examinationPrecision",
        "treatment_overall_score": "treatmentOverallScore",
        "treatment_safety": "treatmentSafety",
        "treatment_effectiveness_alignment": "treatmentEffectivenessAlignment",
        "treatment_personalization": "treatmentPersonalization",
    }
    values = {key: [] for key in fields}
    count = 0
    failed = 0
    pending = 0
    no_exam = []
    recalls = []
    official_precisions = []
    for row in records:
        report = row.get("report", row)
        if not isinstance(report, Mapping):
            continue
        status = report.get("evaluationStatus", report.get("status", "evaluated"))
        if status == "pending":
            pending += 1
            continue
        if status != "evaluated":
            failed += 1
            continue
        count += 1
        for output_name, source_name in fields.items():
            value = report.get(source_name, report.get(output_name))
            if valid_score(value):
                values[output_name].append(float(value))
        detail = report.get("examinationDetail")
        if isinstance(detail, Mapping):
            no_exam.append(not bool(detail.get("ordered")))
            if valid_score(detail.get("coverage")):
                recalls.append(float(detail["coverage"]))
            if detail.get("precisionEligible") is True:
                official_precisions.append(report["examinationPrecision"])
    result = {name: mean(items) if items else None for name, items in values.items()}
    result["evaluated_cases"] = count
    result["failed_evaluations"] = failed
    result["pending_evaluations"] = pending
    result["expected_cases"] = count + failed + pending if expected_count is None else expected_count
    result["evaluation_coverage"] = count / result["expected_cases"] if result["expected_cases"] else 0.0
    result["metric_counts"] = {name: len(items) for name, items in values.items()}
    result["no_examination_rate"] = mean(no_exam) if no_exam else None
    result["examination_recall"] = mean(recalls) if recalls else None
    result["official_examination_precision"] = mean(official_precisions) if official_precisions else None
    # A composite score uses a fixed denominator; missing dimensions never change weights.
    core = [result["diagnosis_accuracy"], result["examination_precision"], result["treatment_overall_score"]]
    core_counts = [len(values[key]) for key in ("diagnosis_accuracy", "examination_precision", "treatment_overall_score")]
    result["overall_clinical_score"] = mean(core) if all(value is not None for value in core) and all(n == count for n in core_counts) else None
    result["metric_missing_counts"] = {name: result["expected_cases"] - len(items) for name, items in values.items()}
    result["complete_clinical_coverage"] = all(n == result["expected_cases"] for n in core_counts)
    return result


def learning_curve_metrics(points: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    checkpoint_rows = sorted(
        (
            row for row in points
            if row.get("split") == "primary_test" and isinstance(row.get("checkpoint"), int)
        ),
        key=lambda row: int(row["checkpoint"]),
    )
    metric_names = (
        "diagnosis_accuracy",
        "examination_precision",
        "treatment_overall_score",
        "overall_clinical_score",
    )
    result: dict[str, Any] = {"checkpoint_count": len(checkpoint_rows)}
    for metric in metric_names:
        samples = [
            (int(row["checkpoint"]), float(row[metric]))
            for row in checkpoint_rows if isinstance(row.get(metric), (int, float))
        ]
        complete = len(samples) == len(checkpoint_rows) and all(
            row.get("metric_counts", {}).get(metric, row.get("expected_cases")) == row.get("expected_cases")
            and row.get("failed_evaluations", 0) == 0 and row.get("pending_evaluations", 0) == 0
            and row.get("evaluation_coverage", 1) == 1 for row in checkpoint_rows
        )
        result[f"{metric}.curve_complete"] = complete
        if len(samples) < 2 or samples[-1][0] == samples[0][0]:
            result[f"{metric}.gain"] = None
            result[f"{metric}.gain_per_100_cases"] = None
            result[f"{metric}.normalized_auc"] = None
            continue
        span = samples[-1][0] - samples[0][0]
        gain = samples[-1][1] - samples[0][1]
        area = sum(
            (right_x - left_x) * (left_y + right_y) / 2
            for (left_x, left_y), (right_x, right_y) in zip(samples, samples[1:])
        )
        result[f"{metric}.gain"] = gain
        result[f"{metric}.gain_per_100_cases"] = gain * 100 / span
        result[f"{metric}.normalized_auc"] = area / span if complete else None
        result[f"{metric}.max_drawdown"] = max(
            max(value for _, value in samples[:index + 1]) - sample[1] for index, sample in enumerate(samples)
        )
    return result


