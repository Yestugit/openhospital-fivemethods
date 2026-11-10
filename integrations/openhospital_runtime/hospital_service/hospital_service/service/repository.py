"""Local JSON data access used by the unified contest service."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

try:
    from .exam.assessment import (
        ExaminationAssessment,
        assess_examination_result,
    )
    from .exam.names import ExaminationNames
except ImportError:
    from exam.assessment import (
        ExaminationAssessment,
        assess_examination_result,
    )
    from exam.names import ExaminationNames

GRAYSCALE_TEST_DATASET_KEY = "grayscale_test"
GRAYSCALE_TEST_SOURCE_DATASET_KEY = "train"
GRAYSCALE_TEST_PATIENT_COUNT = 10


class ServiceDataRepository:
    """Load patient profiles, examination data, and ground truth for one dataset."""

    def __init__(
        self,
        *,
        base_dir: str | Path,
        dataset_key: str,
        normal_examinations_dir: str | Path = "normal_examinations",
        examination_aliases_path: str | Path = "examination_aliases.json",
    ):
        self.base_dir = Path(base_dir)
        self.dataset_key = str(dataset_key)
        source_dataset_key = (
            GRAYSCALE_TEST_SOURCE_DATASET_KEY
            if self.dataset_key == GRAYSCALE_TEST_DATASET_KEY
            else self.dataset_key
        )
        self.patient_profiles_path = self._resolve(Path("patients_data") / source_dataset_key / "profiles.jsonl")
        self.patient_examinations_path = self._resolve(Path("patients_data") / source_dataset_key / "examination_data.json")
        self.ground_truth_path = self._resolve(Path("patients_data") / source_dataset_key / "ground_truth.json")
        self.patient_departments_path = self._resolve(Path("patients_data") / source_dataset_key / "patient_departments.json")
        self.global_patient_departments_path = self._resolve(Path("patients_data") / "patient_departments.json")
        self.normal_examinations_dir = self._resolve(normal_examinations_dir)
        self.examination_aliases_path = self._resolve(examination_aliases_path)
        self.examination_names = ExaminationNames.load(self.examination_aliases_path)

        self.profiles = self._load_profiles(self.patient_profiles_path)
        if self.dataset_key == GRAYSCALE_TEST_DATASET_KEY:
            self.profiles = dict(list(self.profiles.items())[-GRAYSCALE_TEST_PATIENT_COUNT:])
        self.patient_ids = list(self.profiles.keys())
        self.patient_departments = self._load_patient_departments(
            self.patient_departments_path,
            self.global_patient_departments_path,
        )
        self.patient_examinations = self._load_json(self.patient_examinations_path, default={})
        self._validate_patient_examinations(self.patient_examinations, self.patient_examinations_path)
        self.patient_exam_index = self._build_patient_exam_index(self.patient_examinations)
        self.ground_truth = self._load_json(self.ground_truth_path, default={})
        self.normal_examinations = self._load_normal_examinations(self.normal_examinations_dir)
        self.patient_summaries = [
            patient_summary(
                patient_id,
                self.profiles[patient_id],
                self.patient_departments.get(normalize_patient_id(patient_id), []),
            )
            for patient_id in self.patient_ids
        ]
        self.patients_by_department, self.patients_without_department = self._build_patient_department_buckets(
            self.patient_summaries,
            dataset_key=self.dataset_key,
        )

    def get_patient_ids(self) -> list[str]:
        return list(self.patient_ids)

    def get_patient_summaries(self) -> list[dict[str, Any]]:
        return [dict(summary) for summary in self.patient_summaries]

    def get_patients_by_department(self) -> dict[str, list[str]]:
        return {
            department: list(patient_ids)
            for department, patient_ids in self.patients_by_department.items()
        }

    def get_patients_without_department(self) -> list[str]:
        return list(self.patients_without_department)

    def get_profile(self, patient_id: str) -> dict[str, Any]:
        return self.profiles.get(str(patient_id), {})

    def get_ground_truth(self, patient_id: str) -> dict[str, Any]:
        value = self.ground_truth.get(str(patient_id), {})
        return value if isinstance(value, dict) else {}

    def get_patient_exam_map(self, patient_id: str) -> dict[str, Any]:
        return deepcopy(self.patient_exam_index.get(str(patient_id), {}))

    def get_patient_exam_result(self, patient_id: str, item_name: str) -> Any | None:
        canonical_item = self.canonical_examination_name(item_name)
        result = self.patient_exam_index.get(str(patient_id), {}).get(canonical_item)
        return deepcopy(result)

    def get_patient_exam_assessment(
        self,
        patient_id: str,
        item_name: str,
        result: Any,
    ) -> ExaminationAssessment:
        normal_result = self.get_normal_exam_result(patient_id, item_name)
        return assess_examination_result(result, normal_result)

    def get_examinations(self, patient_id: str) -> list[dict[str, Any]]:
        rows = self.patient_examinations.get(str(patient_id), [])
        return normalize_examination_rows(rows)

    def get_patient_context(self, patient_id: str) -> dict[str, Any]:
        profile = dict(self.get_profile(patient_id))
        profile["ground_truth"] = self.get_ground_truth(patient_id)
        profile["request_examinations"] = self.get_examinations(patient_id)
        return profile

    def get_normal_exam_result(self, patient_id: str, item_name: str) -> Any | None:
        profile = self.get_profile(patient_id)
        age_group = infer_age_group(profile)
        gender = normalize_gender(profile)
        canonical_item = self.canonical_examination_name(item_name)
        result = self.normal_examinations.get(age_group, {}).get(gender, {}).get(canonical_item)
        return deepcopy(result)

    def canonical_examination_name(self, item_name: str) -> str:
        return self.examination_names.canonical(str(item_name))

    def canonicalize_examination_items(self, items: Iterable[str]) -> list[str]:
        return self.examination_names.canonicalize_many(items)

    def get_patient_age_group(self, patient_id: str) -> str:
        return infer_age_group(self.get_profile(patient_id))

    def get_patient_gender(self, patient_id: str) -> str:
        return normalize_gender(self.get_profile(patient_id))

    def _resolve(self, raw_path: str | Path) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return self.base_dir / path

    @staticmethod
    def _load_profiles(path: Path) -> dict[str, dict[str, Any]]:
        profiles: dict[str, dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                profile = json.loads(line)
                patient_id = profile.get("id") or profile.get("patient_id")
                if not patient_id:
                    raise ValueError(f"{path}:{line_number} missing patient id")
                profiles[str(patient_id)] = profile
        if not profiles:
            raise ValueError(f"No patient profiles found in {path}")
        return profiles

    @staticmethod
    def _load_json(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @classmethod
    def _load_patient_departments(cls, dataset_path: Path, global_path: Path) -> dict[str, list[str]]:
        payload: Any = {}
        if dataset_path.exists():
            payload = cls._load_json(dataset_path, default={})
        elif global_path.exists():
            payload = cls._load_json(global_path, default={})
        if not isinstance(payload, dict):
            return {}
        return {
            normalize_patient_id(str(patient_id)): normalize_departments(departments)
            for patient_id, departments in payload.items()
            if str(patient_id).strip() and normalize_departments(departments)
        }

    def _validate_patient_examinations(self, payload: Any, path: Path) -> None:
        if not isinstance(payload, dict):
            return
        for patient_id, rows in payload.items():
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                item_name = str(row.get("item_name") or row.get("name") or "").strip()
                self._validate_canonical_examination(item_name, f"{path}:{patient_id}")

    def _validate_canonical_examination(self, item_name: str, context: str) -> None:
        canonical = self.canonical_examination_name(item_name)
        if canonical != item_name:
            raise ValueError(f"{context}: noncanonical examination {item_name!r}; expected {canonical!r}")

    @classmethod
    def _build_patient_department_buckets(
        cls,
        patient_summaries: list[dict[str, Any]],
        *,
        dataset_key: str,
    ) -> tuple[dict[str, list[str]], list[str]]:
        buckets: dict[str, list[str]] = {}
        unbucketed: list[str] = []
        for summary in patient_summaries:
            patient_id = str(summary.get("patientId") or "").strip()
            if not patient_id:
                continue
            departments = normalize_departments(summary.get("departments"))
            if not departments:
                unbucketed.append(patient_id)
                continue
            department = cls._department_bucket_for_patient(patient_id, departments, dataset_key=dataset_key)
            buckets.setdefault(department, []).append(patient_id)
        return buckets, unbucketed

    @staticmethod
    def _department_bucket_for_patient(patient_id: str, departments: list[str], *, dataset_key: str) -> str:
        if len(departments) == 1:
            return departments[0]
        seed_source = f"{dataset_key}|{patient_id}|{'|'.join(departments)}"
        digest = hashlib.sha256(seed_source.encode("utf-8")).hexdigest()
        return departments[int(digest[:8], 16) % len(departments)]

    @classmethod
    def _build_patient_exam_index(cls, payload: Any) -> dict[str, dict[str, Any]]:
        if not isinstance(payload, dict):
            return {}
        return {
            str(patient_id): mapped
            for patient_id, rows in payload.items()
            if (mapped := cls._map_patient_examination_rows(rows))
        }

    @staticmethod
    def _map_patient_examination_rows(rows: Any) -> dict[str, Any]:
        if isinstance(rows, dict):
            return {str(item): normalize_exam_result(value) for item, value in rows.items()}
        if not isinstance(rows, list):
            return {}

        mapped: dict[str, Any] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            item_name = row.get("item_name") or row.get("name")
            result = row.get("result")
            if item_name and result is not None:
                mapped[str(item_name)] = normalize_exam_result(result)
        return mapped

    def _load_normal_examinations(self, directory: Path) -> dict[str, dict[str, dict[str, Any]]]:
        normal: dict[str, dict[str, dict[str, Any]]] = {}
        for path in sorted(directory.glob("examinations_normal_*.json")):
            age_group = path.stem.replace("examinations_normal_", "")
            payload = self._load_json(path, default={})
            rows = payload.get("results") if isinstance(payload, dict) else None
            if not isinstance(rows, list):
                continue
            normal[age_group] = {"male": {}, "female": {}}
            for row in rows:
                if not isinstance(row, dict):
                    continue
                item_name = row.get("item_name")
                if not is_valid_exam_item_name(item_name):
                    continue
                item_name = str(item_name)
                self._validate_loaded_normal_item_name(item_name, path)
                male_result = row.get("male_result")
                female_result = row.get("female_result")
                if male_result is not None:
                    normal[age_group]["male"][item_name] = normalize_exam_result(male_result)
                if female_result is not None:
                    normal[age_group]["female"][item_name] = normalize_exam_result(female_result)
        return normal

    def _validate_loaded_normal_item_name(self, item_name: str, path: Path) -> None:
        self._validate_canonical_examination(item_name, str(path))

def normalize_gender(profile: dict[str, Any]) -> str:
    demographics = profile.get("demographics") or {}
    gender = str(demographics.get("gender", "")).strip().lower()
    if gender in {"女", "女性", "female", "f", "woman", "girl"}:
        return "female"
    if gender in {"男", "男性", "male", "m", "man", "boy"}:
        return "male"
    return "male"


def display_gender(value: Any) -> str:
    gender = str(value or "").strip().lower()
    if gender in {"女", "女性", "female", "f", "woman", "girl"}:
        return "女"
    if gender in {"男", "男性", "male", "m", "man", "boy"}:
        return "男"
    return str(value or "").strip()


def normalize_patient_id(patient_id: str) -> str:
    return patient_id.strip().replace("_", "-")


def normalize_departments(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def patient_summary(patient_id: str, profile: dict[str, Any], departments: list[str] | None = None) -> dict[str, Any]:
    demographics = profile.get("demographics") if isinstance(profile.get("demographics"), dict) else {}
    name = str(demographics.get("name") or "").strip()
    resolved_departments = normalize_departments(profile.get("department")) or list(departments or [])
    return {
        "patientId": patient_id,
        "patientName": name,
        "age": demographics.get("age"),
        "gender": display_gender(demographics.get("gender")),
        "departments": resolved_departments,
    }


def infer_age_group(profile: dict[str, Any]) -> str:
    demographics = profile.get("demographics") or {}
    try:
        age = float(demographics.get("age"))
    except (TypeError, ValueError):
        return "adult"
    if age < 2:
        return "infant"
    if age < 12:
        return "child"
    if age < 18:
        return "adolescent"
    if age < 65:
        return "adult"
    return "elderly"


def normalize_items(items: Iterable[Any]) -> list[str]:
    return [str(item).strip() for item in items if str(item).strip()]


def is_valid_exam_item_name(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip()
    if not normalized:
        return False
    if normalized.isdigit():
        return False
    return True


def exam_result_to_string(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("result"))
    return str(value)


def normalize_exam_result(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): normalize_exam_result(nested_value)
            for key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [normalize_exam_result(item) for item in value]
    return str(value)


def normalize_examination_rows(rows: Any) -> list[dict[str, Any]]:
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    if isinstance(rows, dict):
        return [
            {
                "item_name": str(item),
                "result": exam_result_to_string(value),
            }
            for item, value in rows.items()
        ]
    return []
