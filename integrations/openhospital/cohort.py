from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence
from mematlas.core.provenance import file_sha256


SPLITS = ("train", "primary_test", "advanced_test")

class OpenHospitalCohort:
    """Validated single-department patient IDs for the official data splits."""

    def __init__(
        self,
        mapping_path: str | Path,
        *,
        department: str = "心内科",
        patients_data_dir: str | Path | None = None,
    ) -> None:
        self.mapping_path = Path(mapping_path).expanduser().resolve()
        payload = json.loads(self.mapping_path.read_text(encoding="utf-8"))
        try:
            raw = payload["departments"][department]
        except KeyError as exc:
            raise ValueError(f"Department {department!r} is absent from {self.mapping_path}") from exc
        self.department = department
        self.meta = dict(payload.get("meta", {}))
        self._ids = {split: tuple(str(item) for item in raw.get(split, [])) for split in SPLITS}
        self.patients_data_dir = Path(patients_data_dir).expanduser().resolve() if patients_data_dir else None
        self._validate()

    def _validate(self) -> None:
        seen: set[str] = set()
        for split, patient_ids in self._ids.items():
            if len(patient_ids) != len(set(patient_ids)):
                raise ValueError(f"Duplicate patient ID in {split}")
            overlap = seen.intersection(patient_ids)
            if overlap:
                raise ValueError(f"Patient IDs cross data splits: {sorted(overlap)[:3]}")
            seen.update(patient_ids)
            if self.patients_data_dir is None:
                continue
            split_dir = self.patients_data_dir / split
            departments = json.loads((split_dir / "patient_departments.json").read_text(encoding="utf-8"))
            profiles = {
                str(row["id"])
                for line in (split_dir / "profiles.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip() and (row := json.loads(line))
            }
            truth_path = split_dir / "ground_truth.json"
            ground_truth = json.loads(truth_path.read_text(encoding="utf-8")) if truth_path.exists() else None
            for patient_id in patient_ids:
                if patient_id not in profiles:
                    raise ValueError(f"{patient_id} is absent from {split}/profiles.jsonl")
                labels = departments.get(patient_id.replace("_", "-"))
                if labels != [self.department]:
                    raise ValueError(f"{patient_id} has non-exclusive department labels: {labels!r}")
                if ground_truth is not None:
                    diagnosis = ground_truth.get(patient_id, {}).get("final_diagnosis")
                    if isinstance(diagnosis, str):
                        diagnosis = [diagnosis]
                    if not isinstance(diagnosis, list) or not diagnosis or any(
                            not isinstance(value, str) or not value.strip() for value in diagnosis):
                        raise ValueError(f"{patient_id} has incomplete reference diagnosis labels")

    def ids(self, split: str) -> tuple[str, ...]:
        try:
            return self._ids[split]
        except KeyError as exc:
            raise ValueError(f"Unknown OpenHospital split: {split}") from exc

    def freeze(
        self,
        output_path: str | Path,
        *,
        selected: Mapping[str, Sequence[str]] | None = None,
    ) -> Path:
        frozen: dict[str, list[str]] = {}
        for split in SPLITS:
            values = list((selected or {}).get(split, self.ids(split)))
            allowed = set(self.ids(split))
            invalid = [value for value in values if value not in allowed]
            if invalid:
                raise ValueError(f"{split} includes invalid or cross-department IDs: {invalid[:3]}")
            if len(values) != len(set(values)):
                raise ValueError(f"{split} includes duplicate IDs")
            frozen[split] = values
        if set(frozen["train"]) & (set(frozen["primary_test"]) | set(frozen["advanced_test"])):
            raise ValueError("Frozen train and test IDs overlap")
        source_hashes: dict[str, str] = {"mapping": file_sha256(self.mapping_path)}
        if self.patients_data_dir:
            for split in SPLITS:
                for name in ("profiles.jsonl", "patient_departments.json", "ground_truth.json"):
                    path = self.patients_data_dir / split / name
                    if path.is_file():
                        source_hashes[f"{split}/{name}"] = file_sha256(path)
        payload = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "department": self.department,
            "selection_rule": "department label list must equal [department]",
            "splits": frozen,
            "counts": {key: len(value) for key, value in frozen.items()},
            "source_hashes": source_hashes,
        }
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return output


