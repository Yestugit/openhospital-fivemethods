#!/usr/bin/env python3
"""Extract patient-to-department mappings from generated JSONL patient data."""

from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = SCRIPT_DIR.parents[3] / "data_zh" / "patients_data_name_ZH"
DEFAULT_PATIENTS_DATA_DIR = SCRIPT_DIR.parent / "data" / "patients_data"
DEFAULT_OUTPUT_PATH = DEFAULT_PATIENTS_DATA_DIR / "patient_departments.json"


def normalize_patient_id(patient_id: str) -> str:
    return patient_id.strip().replace("_", "-")


def text_value(value: Any) -> str:
    return str(value or "").strip()


def normalize_departments(value: Any) -> list[str]:
    if isinstance(value, list):
        return unique_text_values(value)

    text = text_value(value)
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, list):
            return unique_text_values(parsed)
    return [text]


def unique_text_values(values: list[Any]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = text_value(value)
        if not text or text in seen:
            continue
        normalized.append(text)
        seen.add(text)
    return normalized


def read_jsonl_patient_departments(source_dir: Path) -> dict[str, list[str]]:
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)

    patient_departments: dict[str, list[str]] = {}
    raw_ids_by_normalized_id: dict[str, str] = {}
    for source_path in sorted(source_dir.glob("*.jsonl")):
        with source_path.open(encoding="utf-8") as source:
            for line_no, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                patient_id = text_value(row.get("patient_id") or row.get("id"))
                departments = normalize_departments(row.get("department"))
                if not patient_id:
                    raise ValueError(f"{source_path}:{line_no} missing patient_id")
                if not departments:
                    raise ValueError(f"{source_path}:{line_no} missing department for {patient_id}")

                normalized_id = normalize_patient_id(patient_id)
                previous_departments = patient_departments.get(normalized_id)
                if previous_departments and previous_departments != departments:
                    raw_id = raw_ids_by_normalized_id.get(normalized_id, normalized_id)
                    raise ValueError(
                        f"Conflicting departments for {raw_id}: "
                        f"{previous_departments!r} vs {departments!r} in {source_path}:{line_no}"
                    )

                patient_departments[normalized_id] = departments
                raw_ids_by_normalized_id[normalized_id] = patient_id

    if not patient_departments:
        raise ValueError(f"No patient rows found under {source_dir}")
    return patient_departments


def write_json(path: Path, payload: Any, *, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        json.dump(payload, target, ensure_ascii=False, indent=2)
        target.write("\n")


def invert_department_mapping(patient_departments: dict[str, list[str]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for patient_id, departments in patient_departments.items():
        for department in departments:
            grouped[department].append(patient_id)
    return {
        department: sorted(patient_ids)
        for department, patient_ids in sorted(grouped.items(), key=lambda item: item[0])
    }


def read_dataset_patient_ids(dataset_dir: Path) -> list[str]:
    profiles_path = dataset_dir / "profiles.jsonl"
    if not profiles_path.exists():
        return []

    patient_ids: list[str] = []
    with profiles_path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            patient_id = text_value(row.get("id") or row.get("patient_id"))
            if not patient_id:
                raise ValueError(f"{profiles_path}:{line_no} missing patient id")
            patient_ids.append(normalize_patient_id(patient_id))
    return patient_ids


def write_dataset_department_files(
    patients_data_dir: Path,
    patient_departments: dict[str, list[str]],
    *,
    dry_run: bool,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for dataset_dir in sorted(path for path in patients_data_dir.iterdir() if path.is_dir()):
        patient_ids = read_dataset_patient_ids(dataset_dir)
        if not patient_ids:
            continue

        dataset_mapping = {
            patient_id: patient_departments[patient_id]
            for patient_id in patient_ids
            if patient_id in patient_departments
        }
        if not dataset_mapping:
            continue

        write_json(dataset_dir / "patient_departments.json", dataset_mapping, dry_run=dry_run)
        counts[dataset_dir.name] = len(dataset_mapping)
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract each patient's department from patients_data_name_ZH JSONL files "
            "and write mappings used by carbon train/test patient filtering."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"Directory containing source *.jsonl files. Default: {DEFAULT_SOURCE_DIR}",
    )
    parser.add_argument(
        "--patients-data-dir",
        type=Path,
        default=DEFAULT_PATIENTS_DATA_DIR,
        help=f"patients_data root used for per-dataset mapping files. Default: {DEFAULT_PATIENTS_DATA_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Global patient_departments.json output path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--skip-dataset-files",
        action="store_true",
        help="Only write the global mapping file, not train/test dataset-specific files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print counts without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    patients_data_dir = args.patients_data_dir.resolve()
    output_path = args.output.resolve()

    patient_departments = read_jsonl_patient_departments(source_dir)
    departments = invert_department_mapping(patient_departments)

    write_json(output_path, patient_departments, dry_run=args.dry_run)
    department_output = output_path.with_name("patients_by_department.json")
    write_json(department_output, departments, dry_run=args.dry_run)

    dataset_counts: dict[str, int] = {}
    if not args.skip_dataset_files and patients_data_dir.exists():
        dataset_counts = write_dataset_department_files(
            patients_data_dir,
            patient_departments,
            dry_run=args.dry_run,
        )

    action = "Validated" if args.dry_run else "Wrote"
    print(f"{action} {len(patient_departments)} patient department mappings.")
    print(f"Departments: {len(departments)}")
    print(f"Global mapping: {output_path}")
    print(f"Grouped mapping: {department_output}")
    if dataset_counts:
        print("Dataset files:")
        for dataset_key, count in dataset_counts.items():
            print(f"  {dataset_key}: {count}")


if __name__ == "__main__":
    main()
