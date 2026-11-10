from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from mematlas.core.evaluation import EvaluationResult


class ReportWriter:
    _SUMMARY_METRICS = (
        "common.exact_match",
        "common.token_f1",
        "native.qa_f1",
        "native.judge_accuracy",
        "overall_clinical_score",
        "diagnosis_accuracy",
        "examination_precision",
        "treatment_overall_score",
        "evaluation_coverage",
    )
    _SUMMARY_RESOURCES = (
        "common.memory_construction_time_s",
        "common.memory_learning_time_s",
        "common.retrieval_mean_ms",
        "common.response_mean_s",
        "common.total_tokens",
        "common.process_peak_rss_mib",
        "common.memory_storage_mib",
    )
    _RESOURCE_LABELS = {
        "common.memory_construction_time_s": ("Time efficiency", "Memory construction time", "s"),
        "common.memory_learning_time_s": ("Time efficiency", "Memory learning time", "s"),
        "common.retrieval_mean_ms": ("Time efficiency", "Mean retrieval latency", "ms/request"),
        "common.response_mean_s": ("Time efficiency", "Mean question response time", "s/question"),
        "common.judging_time_s": ("Time efficiency", "Total scoring time", "s"),
        "common.total_tokens": ("Model usage", "Memory + answer model tokens", "tokens"),
        "common.total_evaluation_tokens": ("Model usage", "All recorded model tokens including judges", "tokens"),
        "common.llm_calls": ("Model usage", "Memory + answer LLM requests", "requests"),
        "common.embedding_calls": ("Model usage", "Memory + answer embedding requests", "requests"),
        "common.process_peak_rss_mib": ("Space overhead", "Sampled peak process-tree RSS", "MiB"),
        "common.memory_storage_mib": ("Space overhead", "Live memory data storage", "MiB"),
    }

    @classmethod
    def _resource_labels(cls):
        labels = dict(cls._RESOURCE_LABELS)
        for role in ("memory", "agent", "judge"):
            for key, label, unit in (("prompt_tokens", "input tokens", "tokens"),
                                     ("completion_tokens", "output tokens", "tokens"),
                                     ("llm_calls", "LLM requests", "requests"),
                                     ("embedding_calls", "embedding requests", "requests")):
                labels[f"common.{role}_{key}"] = ("Model usage", f"{role.title()} {label}", unit)
        return labels

    @staticmethod
    def _display(value) -> str:
        if value is None:
            return "unknown"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    def write(self, results: Sequence[EvaluationResult], output_dir: str | Path) -> dict[str, Path]:
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "results.json"
        markdown_path = output_dir / "report.md"
        json_path.write_text(
            json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        available_metrics = {name for result in results for name in result.metrics}
        available_resources = {name for result in results for name in result.resource_metrics}
        metric_names = [name for name in self._SUMMARY_METRICS if name in available_metrics]
        resource_names = [name for name in self._SUMMARY_RESOURCES if name in available_resources]
        labels = self._resource_labels()
        resource_columns = [f"{labels[name][1]} ({labels[name][2]})" for name in resource_names]
        columns = ["Method", "Implementation", "Mode", "Benchmark", "Predictions", *metric_names, *resource_columns, "Failures"]
        lines = [
            "# MemAtlas Evaluation Report",
            "",
            "## Comparison",
            "",
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for result in results:
            method_config = result.run_metadata.get("config", {}).get("method", {})
            mode = method_config.get("params", {}).get("implementation_mode", "standard")
            fidelity = (result.run_metadata.get("method") or {}).get("fidelity", "unknown")
            values = [result.method_id, fidelity, mode, result.benchmark_id, str(len(result.predictions))]
            values.extend(self._display(result.metrics.get(name)) for name in metric_names)
            values.extend(self._display(result.resource_metrics.get(name)) for name in resource_names)
            values.append(str(len(result.failures)))
            lines.append("| " + " | ".join(values) + " |")
        for index, result in enumerate(results, start=1):
            run_id = str(result.run_metadata.get("run_id") or "unknown")
            lines.extend([
                "",
                f"## Run {index}: {result.method_id}",
                "",
                f"Benchmark: `{result.benchmark_id}`  ",
                f"Run ID: `{run_id}`  ",
                f"Predictions: {len(result.predictions)}  ",
                f"Failures: {len(result.failures)}",
                "",
                "### Quality Metrics",
                "",
                "| Metric | Value |",
                "| --- | --- |",
            ])
            lines.extend(
                f"| {name} | {self._display(value)} |"
                for name, value in sorted(result.metrics.items())
            )
            if result.learning_curve:
                lines.extend(["", "### Learning Progress", "", "| Field | Value |", "| --- | --- |"])
                for category in ("learning_curve_summary", "progress"):
                    lines.extend(f"| {name} | {self._display(value)} |"
                                 for name, value in result.run_metadata.get(category, {}).items())
            lines.extend([
                "",
                "### Efficiency and Resource Measurements",
                "",
                "| Dimension | Metric | Unit | Value |",
                "| --- | --- | --- | --- |",
            ])
            lines.extend(
                f"| {labels[name][0]} | {labels[name][1]} | {labels[name][2]} | {self._display(value)} |"
                for name, value in sorted(result.resource_metrics.items())
                if name in labels
            )
            lines.extend(["", "### Run Diagnostics", "", "| Field | Value |", "| --- | --- |"])
            lines.extend(f"| {name} | {self._display(value)} |"
                         for name, value in sorted(result.resource_metrics.items()) if name not in labels)
            if result.failures:
                lines.extend([
                    "",
                    "### Failures",
                    "",
                    "| Stage | Case | Type | Message |",
                    "| --- | --- | --- | --- |",
                ])
                for failure in result.failures:
                    lines.append("| " + " | ".join([
                        self._display(failure.get("stage")),
                        self._display(failure.get("case_id")),
                        self._display(failure.get("error_type")),
                        self._display(failure.get("message")),
                    ]) + " |")
        markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        for result in results:
            run_id = str(result.run_metadata.get("run_id") or "unknown")
            method_dir = output_dir / result.method_id / run_id
            method_dir.mkdir(parents=True, exist_ok=True)
            if result.learning_curve:
                (method_dir / "curve.json").write_text(
                    json.dumps(list(result.learning_curve), ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8",
                )
            (method_dir / "predictions.jsonl").write_text(
                "".join(
                    json.dumps(item.to_dict(), ensure_ascii=False, default=str) + "\n"
                    for item in result.predictions
                ),
                encoding="utf-8",
            )
            (method_dir / "traces.jsonl").write_text(
                "".join(
                    json.dumps(item, ensure_ascii=False, default=str) + "\n"
                    for item in result.traces
                ),
                encoding="utf-8",
            )
            (method_dir / "manifest.json").write_text(
                json.dumps(result.run_metadata, ensure_ascii=False, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
        return {"json": json_path, "markdown": markdown_path}
