"""Import existing static examination results into the shared result store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..config import load_settings
from ..exam.result_store import (
    ExaminationResultRecord,
    ExaminationResultStore,
    InMemoryExaminationResultStore,
    PostgresExaminationResultStore,
)
from ..repository import ServiceDataRepository

DATASETS = ("train", "primary_test", "advanced_test", "all")


def import_repository_results(
    repository: ServiceDataRepository,
    store: ExaminationResultStore,
) -> dict[str, int]:
    inserted = 0
    preserved = 0
    for patient_id, examinations in repository.patient_exam_index.items():
        for examination, result in examinations.items():
            existing = store.get(repository.dataset_key, patient_id, examination)
            if existing is not None:
                preserved += 1
                continue
            assessment = repository.get_patient_exam_assessment(patient_id, examination, result)
            record = ExaminationResultRecord(
                dataset_key=repository.dataset_key,
                patient_id=patient_id,
                examination_name=examination,
                result=result,
                indicator_statuses=assessment.indicator_statuses,
                status=assessment.status,
                abnormal_indicators=assessment.abnormal_indicators,
                source="existing",
            )
            stored = store.put_if_absent(record)
            if stored == record:
                inserted += 1
            else:
                preserved += 1
    return {"inserted": inserted, "preserved": preserved}


def import_datasets(
    *,
    data_root: Path,
    datasets: list[str],
    store: ExaminationResultStore,
) -> dict[str, Any]:
    reports: dict[str, dict[str, int]] = {}
    for dataset in datasets:
        repository = ServiceDataRepository(base_dir=data_root, dataset_key=dataset)
        reports[dataset] = import_repository_results(repository, store)
    return {
        "datasets": reports,
        "inserted": sum(report["inserted"] for report in reports.values()),
        "preserved": sum(report["preserved"] for report in reports.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", choices=DATASETS, dest="datasets")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    settings = load_settings()
    datasets = args.datasets or list(DATASETS)
    store: ExaminationResultStore
    if args.dry_run:
        store = InMemoryExaminationResultStore()
    else:
        store = PostgresExaminationResultStore(
            settings.pg_dsn,
            pool_max_size=settings.db_pool_max_size,
        )
    try:
        report = import_datasets(
            data_root=Path(settings.data_dir),
            datasets=datasets,
            store=store,
        )
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()
    report["dry_run"] = bool(args.dry_run)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
