"""Normalize persisted indicator names and build examination_indicators.json."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..exam.assessment import assess_examination_result
from .examination_terminology_rules import (
    EXAMINATION_ALIASES,
    REVIEWED_INDICATOR_CORRECTIONS,
)

AGE_GROUPS = ("infant", "child", "adolescent", "adult", "elderly")
DATASETS = ("train", "primary_test", "advanced_test", "all")
SUPPLEMENTAL_DESCRIPTION_KEY = "补充描述"
LATIN_ABBREVIATION = re.compile(r"[A-Za-z][A-Za-z0-9.+/-]{0,20}")
CHINESE = re.compile(r"[\u3400-\u9fff]")


@dataclass
class IndicatorAggregate:
    age_groups: set[str] = field(default_factory=set)
    genders: set[str] = field(default_factory=set)
    seen_in_normal: bool = False
    seen_in_normal_patient: bool = False
    seen_in_abnormal_patient: bool = False


def canonical_examination_name(value: Any) -> str:
    text = str(value or "").strip()
    return EXAMINATION_ALIASES.get(text, text)


def normalize_indicator_name(value: Any) -> str:
    text = _structural_indicator_name(value)
    return _indicator_corrections().get(text, text)


def _structural_indicator_name(value: Any) -> str:
    text = " ".join(unicodedata.normalize("NFKC", str(value or "")).split())
    match = re.fullmatch(r"(.+?)\(([^()]*)\)", text)
    if match:
        left, inside = (part.strip() for part in match.groups())
        if CHINESE.search(left) and LATIN_ABBREVIATION.fullmatch(inside):
            return left
        if LATIN_ABBREVIATION.fullmatch(left) and CHINESE.search(inside):
            return inside
    return text


_CORRECTIONS: dict[str, str] | None = None


def _indicator_corrections() -> dict[str, str]:
    global _CORRECTIONS
    if _CORRECTIONS is not None:
        return _CORRECTIONS
    corrections = {
        _structural_indicator_name(alias): _structural_indicator_name(canonical)
        for alias, canonical in REVIEWED_INDICATOR_CORRECTIONS.items()
    }
    _CORRECTIONS = corrections
    return corrections


class ExaminationIndicatorBuilder:
    def __init__(self, repository_root: Path):
        self.repository_root = repository_root.resolve()
        self.data_root = self.repository_root / "hospital_service/service/data"
        self.catalog_path = self.data_root / "examination_indicators.json"
        self.changed_files: list[str] = []
        self.aggregates: dict[str, dict[str, IndicatorAggregate]] = {}
        self.normal_results: dict[tuple[str, str, str], dict[str, Any]] = {}

    def build(self, *, write: bool) -> dict[str, Any]:
        normal_payloads = self._process_normal_templates(write=write)
        patient_payloads = self._process_patient_results(write=write)
        catalog = self._catalog_payload()
        catalog_changed = _json_changed(self.catalog_path, catalog)
        if write and catalog_changed:
            _write_json(self.catalog_path, catalog, newline="\n")
            self.changed_files.append(str(self.catalog_path.relative_to(self.repository_root)))
        self._validate_public_catalogs()
        return {
            "examination_count": len(catalog["examinations"]),
            "indicator_count": sum(
                len(value["indicators"])
                for value in catalog["examinations"].values()
            ),
            "normal_files": normal_payloads,
            "patient_files": patient_payloads,
            "changes_required": len(self.changed_files) if write else int(catalog_changed) + normal_payloads + patient_payloads,
            "changed_files": self.changed_files,
        }

    def _process_normal_templates(self, *, write: bool) -> int:
        changed = 0
        directory = self.data_root / "normal_examinations"
        for path in sorted(directory.glob("examinations_normal_*.json")):
            payload = _load_json(path)
            rows = payload.get("results") if isinstance(payload, dict) else None
            if not isinstance(rows, list):
                raise ValueError(f"{path} must contain a results array")
            age_group = path.stem.removeprefix("examinations_normal_")
            seen_examinations: set[str] = set()
            for row in rows:
                if not isinstance(row, dict):
                    continue
                examination = canonical_examination_name(row.get("item_name"))
                if not examination:
                    continue
                if examination in seen_examinations:
                    raise ValueError(f"Duplicate examination {examination!r} in {path}")
                seen_examinations.add(examination)
                row["item_name"] = examination
                for gender, key in (("male", "male_result"), ("female", "female_result")):
                    result = row.get(key)
                    if not isinstance(result, dict):
                        continue
                    row[key] = self._normalize_result(
                        result,
                        context=f"{path}:{examination}:{gender}",
                    )
                    self.normal_results[(age_group, gender, examination)] = row[key]
                    self._record_result(
                        examination,
                        row[key],
                        age_group=age_group,
                        gender=gender,
                        normal=True,
                        patient_statuses=None,
                    )
            if _json_changed(path, payload):
                changed += 1
                if write:
                    _write_json(path, payload, newline=_newline(path))
                    self.changed_files.append(str(path.relative_to(self.repository_root)))
        return changed

    def _process_patient_results(self, *, write: bool) -> int:
        changed = 0
        for dataset in DATASETS:
            directory = self.data_root / "patients_data" / dataset
            path = directory / "examination_data.json"
            payload = _load_json(path)
            profiles = _load_profiles(directory / "profiles.jsonl")
            if not isinstance(payload, dict):
                raise ValueError(f"{path} must contain a patient object")
            for patient_id, rows in payload.items():
                if not isinstance(rows, list):
                    continue
                seen_examinations: set[str] = set()
                profile = profiles.get(str(patient_id), {})
                age_group = _age_group(profile)
                gender = _gender(profile)
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    key = "item_name" if "item_name" in row else "name"
                    examination = canonical_examination_name(row.get(key))
                    if not examination:
                        continue
                    if examination in seen_examinations:
                        raise ValueError(f"Duplicate examination {examination!r} for {patient_id} in {path}")
                    seen_examinations.add(examination)
                    row[key] = examination
                    result = row.get("result")
                    if not isinstance(result, dict):
                        continue
                    row["result"] = self._normalize_result(
                        result,
                        context=f"{path}:{patient_id}:{examination}",
                    )
                    normal_result = self.normal_results.get((age_group, gender, examination))
                    assessment = assess_examination_result(row["result"], normal_result)
                    self._record_result(
                        examination,
                        row["result"],
                        age_group=age_group,
                        gender=gender,
                        normal=False,
                        patient_statuses=assessment.indicator_statuses,
                    )
            if _json_changed(path, payload):
                changed += 1
                if write:
                    _write_json(path, payload, newline=_newline(path))
                    self.changed_files.append(str(path.relative_to(self.repository_root)))
        return changed

    def _normalize_result(self, result: dict[str, Any], *, context: str) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for raw_name, value in result.items():
            name = str(raw_name) if str(raw_name) == SUPPLEMENTAL_DESCRIPTION_KEY else normalize_indicator_name(raw_name)
            if not name:
                raise ValueError(f"Blank indicator name in {context}")
            if name in normalized and normalized[name] != value:
                raise ValueError(f"Conflicting values for indicator {name!r} in {context}")
            normalized[name] = value
        return normalized

    def _record_result(
        self,
        examination: str,
        result: dict[str, Any],
        *,
        age_group: str,
        gender: str,
        normal: bool,
        patient_statuses: dict[str, str] | None,
    ) -> None:
        indicators = self.aggregates.setdefault(examination, {})
        for name in result:
            if name == SUPPLEMENTAL_DESCRIPTION_KEY:
                continue
            aggregate = indicators.setdefault(name, IndicatorAggregate())
            aggregate.age_groups.add(age_group)
            aggregate.genders.add(gender)
            aggregate.seen_in_normal = aggregate.seen_in_normal or normal
            if not normal and patient_statuses is not None:
                status = patient_statuses.get(name, "abnormal")
                aggregate.seen_in_normal_patient = aggregate.seen_in_normal_patient or status == "normal"
                aggregate.seen_in_abnormal_patient = aggregate.seen_in_abnormal_patient or status == "abnormal"

    def _catalog_payload(self) -> dict[str, Any]:
        return {
            "metadata": {"schema_version": 1},
            "examinations": {
                examination: {
                    "indicators": [
                        {
                            "name": name,
                            "age_groups": sorted(aggregate.age_groups, key=AGE_GROUPS.index),
                            "genders": sorted(aggregate.genders),
                            "abnormal_case_only": (
                                aggregate.seen_in_abnormal_patient
                                and not aggregate.seen_in_normal
                                and not aggregate.seen_in_normal_patient
                            ),
                        }
                        for name, aggregate in sorted(indicators.items())
                    ]
                }
                for examination, indicators in sorted(self.aggregates.items())
            },
        }

    def _validate_public_catalogs(self) -> None:
        paths = (
            self.repository_root / "baseline_example/data/ref_data/examinations_catalog.json",
            self.repository_root / "web_server/backend/app/reference-data/catalogs/examinations_catalog.json",
        )
        expected: set[str] | None = None
        for path in paths:
            payload = _load_json(path)
            rows = _public_catalog_rows(payload)
            if not isinstance(rows, list):
                raise ValueError(f"{path} must contain an examination array")
            names = [canonical_examination_name(row.get("name")) for row in rows if isinstance(row, dict)]
            if len(names) != len(set(names)):
                raise ValueError(f"Duplicate canonical examination name in {path}")
            current = set(names)
            if expected is not None and current != expected:
                raise ValueError("Public examination catalogs contain different canonical names")
            expected = current
        catalog_names = set(self.aggregates)
        if expected is not None and not catalog_names.issubset(expected):
            missing = sorted(catalog_names - expected)
            raise ValueError(f"Indicator catalog examinations missing from public catalog: {missing[:10]}")


def _load_profiles(path: Path) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            profile = json.loads(line)
            patient_id = profile.get("id") or profile.get("patient_id")
            if patient_id:
                profiles[str(patient_id)] = profile
    return profiles


def _public_catalog_rows(payload: Any) -> list[Any] | None:
    value = payload.get("examinations") if isinstance(payload, dict) else payload
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [row for rows in value.values() if isinstance(rows, list) for row in rows]
    return None


def _age_group(profile: dict[str, Any]) -> str:
    demographics = profile.get("demographics") if isinstance(profile.get("demographics"), dict) else {}
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


def _gender(profile: dict[str, Any]) -> str:
    demographics = profile.get("demographics") if isinstance(profile.get("demographics"), dict) else {}
    value = str(demographics.get("gender") or "").strip().lower()
    return "female" if value in {"女", "女性", "female", "f", "woman", "girl"} else "male"


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _json_changed(path: Path, payload: Any) -> bool:
    return not path.exists() or _load_json(path) != payload


def _newline(path: Path) -> str:
    return "\r\n" if path.exists() and b"\r\n" in path.read_bytes() else "\n"


def _write_json(path: Path, payload: Any, *, newline: str) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if newline == "\r\n":
        text = text.replace("\n", "\r\n")
    path.write_text(text, encoding="utf-8", newline="")


def _default_repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--repository-root", type=Path, default=_default_repository_root())
    args = parser.parse_args()
    builder = ExaminationIndicatorBuilder(args.repository_root)
    report = builder.build(write=args.write)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.check and report["changes_required"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
