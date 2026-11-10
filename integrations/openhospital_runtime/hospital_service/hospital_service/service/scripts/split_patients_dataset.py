#!/usr/bin/env python3
"""Split patient data from all/ into train, primary_test, and advanced_test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent / "data" / "patients_data"
DATA_FILES = ("profiles.jsonl", "examination_data.json", "ground_truth.json")


def normalize_patient_id(patient_id: str) -> str:
    """Normalize ID spellings used by txt lists and source data."""
    return patient_id.strip().replace("_", "-")


def read_id_file(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            raw_id = line.strip()
            if not raw_id:
                continue
            normalized_id = normalize_patient_id(raw_id)
            if normalized_id in ids:
                raise ValueError(f"Duplicate patient id in {path}:{line_no}: {raw_id}")
            ids.add(normalized_id)
    return ids


def ensure_output_dirs(data_dir: Path) -> dict[str, Path]:
    split_dirs = {
        "train": data_dir / "train",
        "primary_test": data_dir / "primary_test",
        "advanced_test": data_dir / "advanced_test",
    }
    for path in split_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return split_dirs


def split_profiles(
    source_path: Path,
    output_dirs: dict[str, Path],
    primary_ids: set[str],
    advanced_ids: set[str],
    dry_run: bool,
) -> tuple[dict[str, int], set[str]]:
    counts = {split: 0 for split in output_dirs}
    seen_ids: set[str] = set()

    handles = {}
    try:
        if not dry_run:
            handles = {
                split: (path / "profiles.jsonl").open("w", encoding="utf-8")
                for split, path in output_dirs.items()
            }

        with source_path.open(encoding="utf-8") as source:
            for line_no, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                patient_id = row.get("id")
                if not isinstance(patient_id, str):
                    raise ValueError(f"Missing string id in {source_path}:{line_no}")

                normalized_id = normalize_patient_id(patient_id)
                if normalized_id in seen_ids:
                    raise ValueError(f"Duplicate patient id in profiles: {patient_id}")
                seen_ids.add(normalized_id)

                split = choose_split(normalized_id, primary_ids, advanced_ids)
                counts[split] += 1
                if not dry_run:
                    handles[split].write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        for handle in handles.values():
            handle.close()

    return counts, seen_ids


def split_json_dict(
    source_path: Path,
    output_dirs: dict[str, Path],
    primary_ids: set[str],
    advanced_ids: set[str],
    dry_run: bool,
) -> dict[str, int]:
    with source_path.open(encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level object in {source_path}")

    split_data = {split: {} for split in output_dirs}
    for patient_id, value in data.items():
        normalized_id = normalize_patient_id(patient_id)
        split = choose_split(normalized_id, primary_ids, advanced_ids)
        split_data[split][patient_id] = value

    if not dry_run:
        for split, rows in split_data.items():
            output_path = output_dirs[split] / source_path.name
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)
                f.write("\n")

    return {split: len(rows) for split, rows in split_data.items()}


def choose_split(
    normalized_id: str,
    primary_ids: set[str],
    advanced_ids: set[str],
) -> str:
    if normalized_id in primary_ids:
        return "primary_test"
    if normalized_id in advanced_ids:
        return "advanced_test"
    return "train"


def validate_ids(
    primary_ids: set[str],
    advanced_ids: set[str],
    all_profile_ids: set[str],
) -> None:
    overlap = primary_ids & advanced_ids
    if overlap:
        examples = ", ".join(sorted(overlap)[:10])
        raise ValueError(f"Primary and advanced id files overlap: {examples}")

    missing_primary = primary_ids - all_profile_ids
    missing_advanced = advanced_ids - all_profile_ids
    if missing_primary or missing_advanced:
        messages = []
        if missing_primary:
            messages.append(
                "primary missing: " + ", ".join(sorted(missing_primary)[:10])
            )
        if missing_advanced:
            messages.append(
                "advanced missing: " + ", ".join(sorted(missing_advanced)[:10])
            )
        raise ValueError("Some patient ids were not found in all/profiles.jsonl; " + "; ".join(messages))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split patients_data/all into train, primary_test, and advanced_test "
            "using primary_test_patient_ids.txt and advanced_test_patient_ids.txt."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"patients_data directory. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print split counts without writing output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    all_dir = data_dir / "all"

    for file_name in DATA_FILES:
        source_path = all_dir / file_name
        if not source_path.exists():
            raise FileNotFoundError(source_path)

    primary_ids = read_id_file(data_dir / "primary_test_patient_ids.txt")
    advanced_ids = read_id_file(data_dir / "advanced_test_patient_ids.txt")
    output_dirs = ensure_output_dirs(data_dir)

    profile_counts, all_profile_ids = split_profiles(
        all_dir / "profiles.jsonl",
        output_dirs,
        primary_ids,
        advanced_ids,
        args.dry_run,
    )
    validate_ids(primary_ids, advanced_ids, all_profile_ids)

    json_counts = {
        file_name: split_json_dict(
            all_dir / file_name,
            output_dirs,
            primary_ids,
            advanced_ids,
            args.dry_run,
        )
        for file_name in ("examination_data.json", "ground_truth.json")
    }

    print("Split counts:")
    print(f"  profiles.jsonl: {profile_counts}")
    for file_name, counts in json_counts.items():
        print(f"  {file_name}: {counts}")

    action = "Validated" if args.dry_run else "Wrote"
    print(f"{action} dataset splits under {data_dir}")


if __name__ == "__main__":
    main()
