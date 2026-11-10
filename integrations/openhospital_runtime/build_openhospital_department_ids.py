"""Build department-wise OpenHospital patient IDs from the official split data."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SPLITS = ("train", "primary_test", "advanced_test")


def load_split(split: str, *, patients_data: Path) -> dict[str, list[str]]:
    split_dir = patients_data / split
    raw_mapping = json.loads((split_dir / "patient_departments.json").read_text(encoding="utf-8"))
    profile_ids = {
        profile["id"]
        for line in (split_dir / "profiles.jsonl").read_text(encoding="utf-8").splitlines()
        if (profile := json.loads(line))
    }
    normalized_ids = {patient_id.replace("_", "-") for patient_id in profile_ids}
    if set(raw_mapping) != normalized_ids or len(normalized_ids) != len(profile_ids):
        raise ValueError(f"{split}: department mapping and patient profiles disagree")
    # The HTTP service exposes IDs from profiles.jsonl (underscores), while
    # patient_departments.json uses hyphens for the same patients.
    mapping = {patient_id: raw_mapping[patient_id.replace("_", "-")] for patient_id in profile_ids}
    for patient_id, departments in mapping.items():
        if not isinstance(departments, list) or not departments or len(departments) != len(set(departments)):
            raise ValueError(f"{split}: invalid department labels for {patient_id}")
    return mapping


def build(hospital_root: str | Path | None = None) -> dict:
    root = Path(hospital_root or ROOT).expanduser().resolve()
    patients_data = root / "hospital_service/hospital_service/service/data/patients_data"
    mappings = {split: load_split(split, patients_data=patients_data) for split in SPLITS}
    all_mapping = json.loads((patients_data / "patient_departments.json").read_text(encoding="utf-8"))
    combined = {}
    for split, mapping in mappings.items():
        if set(combined).intersection(mapping):
            raise ValueError(f"{split}: patient IDs overlap another split")
        combined.update(mapping)
    if {patient_id.replace("_", "-"): labels for patient_id, labels in combined.items()} != all_mapping:
        raise ValueError("Per-split department labels disagree with the combined mapping")

    # Keep all same-department cases, including comorbidities within that department.
    # A patient with multiple department labels requires cross-department diagnosis.
    departments = sorted({labels[0] for labels in combined.values() if len(labels) == 1})
    grouped = {}
    for department in departments:
        grouped[department] = {}
        for split, mapping in mappings.items():
            grouped[department][split] = sorted(
                patient_id for patient_id, labels in mapping.items() if labels == [department]
            )

    included_counts = {
        split: sum(len(grouped[department][split]) for department in departments)
        for split in SPLITS
    }
    excluded_counts = {
        split: sum(len(labels) > 1 for labels in mapping.values())
        for split, mapping in mappings.items()
    }
    for split, mapping in mappings.items():
        if included_counts[split] + excluded_counts[split] != len(mapping):
            raise ValueError(f"{split}: filtering lost or duplicated patients")

    return {
        "meta": {
            "source": "hospital_service/hospital_service/service/data/patients_data",
            "source_split_counts": {split: len(mapping) for split, mapping in mappings.items()},
            "included_split_counts": included_counts,
            "excluded_cross_department_counts": excluded_counts,
            "department_count": len(departments),
            "id_format": "IDs match profiles.jsonl and the hospital service /patients response (underscores).",
            "selection_rule": "Include a patient only when its department label list contains exactly one department. Same-department comorbidities remain included.",
        },
        "departments": grouped,
    }


def write_department_ids(hospital_root: str | Path | None = None, *, output: str | Path | None = None) -> Path:
    root = Path(hospital_root or ROOT).expanduser().resolve()
    result = build(root)
    target = Path(output).expanduser().resolve() if output else root / "department_patient_ids.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {target} ({result['meta']['department_count']} departments)")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hospital-root", default=str(ROOT))
    parser.add_argument("--output")
    args = parser.parse_args()
    write_department_ids(args.hospital_root, output=args.output)
