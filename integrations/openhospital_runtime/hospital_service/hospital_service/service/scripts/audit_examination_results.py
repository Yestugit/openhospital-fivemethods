"""Audit canonical examination names and indicator catalog consistency."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from ..exam.catalog import ExaminationIndicatorCatalog
from .build_examination_indicators import (
    DATASETS,
    ExaminationIndicatorBuilder,
    canonical_examination_name,
    normalize_indicator_name,
)

VIOLATION_NAMES = (
    "catalog_changes_required",
    "duplicate_patient_examinations",
    "persisted_examination_aliases",
    "duplicate_indicators_within_examination",
    "noncanonical_indicator_names",
    "catalog_indicator_mismatches",
)


def audit_examination_results(repository_root: Path) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    data_root = repository_root / "hospital_service/service/data"
    violations = {name: 0 for name in VIOLATION_NAMES}
    samples: dict[str, list[str]] = {name: [] for name in VIOLATION_NAMES}
    information: Counter[str] = Counter()

    build_report = ExaminationIndicatorBuilder(repository_root).build(write=False)
    if build_report["changes_required"]:
        _record(
            violations,
            samples,
            "catalog_changes_required",
            str(build_report["changes_required"]),
        )
    catalog = ExaminationIndicatorCatalog.load(data_root / "examination_indicators.json")

    for dataset in DATASETS:
        path = data_root / "patients_data" / dataset / "examination_data.json"
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        for patient_id, rows in payload.items():
            if not isinstance(rows, list):
                continue
            information["patients"] += 1
            names = [
                str(row.get("item_name") or row.get("name") or "").strip()
                for row in rows
                if isinstance(row, dict)
            ]
            for examination, count in Counter(names).items():
                if examination and count > 1:
                    _record(
                        violations,
                        samples,
                        "duplicate_patient_examinations",
                        f"{dataset}:{patient_id}:{examination}",
                    )
            for row in rows:
                if not isinstance(row, dict):
                    continue
                examination = str(row.get("item_name") or row.get("name") or "").strip()
                if not examination:
                    continue
                information["examination_rows"] += 1
                if canonical_examination_name(examination) != examination:
                    _record(
                        violations,
                        samples,
                        "persisted_examination_aliases",
                        f"{dataset}:{patient_id}:{examination}",
                    )
                result = row.get("result")
                if not isinstance(result, dict):
                    continue
                indicator_names = [str(name) for name in result if str(name) != "补充描述"]
                for indicator, count in Counter(indicator_names).items():
                    if count > 1:
                        _record(
                            violations,
                            samples,
                            "duplicate_indicators_within_examination",
                            f"{dataset}:{patient_id}:{examination}:{indicator}",
                        )
                    if normalize_indicator_name(indicator) != indicator:
                        _record(
                            violations,
                            samples,
                            "noncanonical_indicator_names",
                            f"{dataset}:{patient_id}:{examination}:{indicator}",
                        )
                catalog_names = {item.name for item in catalog.indicators(examination)}
                for indicator in set(indicator_names) - catalog_names:
                    _record(
                        violations,
                        samples,
                        "catalog_indicator_mismatches",
                        f"{dataset}:{patient_id}:{examination}:{indicator}",
                    )

    return {
        "violations": violations,
        "samples": {name: values for name, values in samples.items() if values},
        "information": dict(information),
    }


def _record(
    violations: dict[str, int],
    samples: dict[str, list[str]],
    name: str,
    sample: str,
) -> None:
    violations[name] += 1
    if len(samples[name]) < 20:
        samples[name].append(sample)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    report = audit_examination_results(repository_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if any(report["violations"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
