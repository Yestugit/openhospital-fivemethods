"""Runtime examination-to-indicator catalog."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

AGE_GROUPS = frozenset({"infant", "child", "adolescent", "adult", "elderly"})
GENDERS = frozenset({"male", "female"})
INDICATOR_FIELDS = frozenset({"name", "age_groups", "genders", "abnormal_case_only"})


@dataclass(frozen=True)
class ExaminationIndicator:
    name: str
    age_groups: tuple[str, ...]
    genders: tuple[str, ...]
    abnormal_case_only: bool


class ExaminationIndicatorCatalog:
    def __init__(self, examinations: dict[str, tuple[ExaminationIndicator, ...]]):
        self._examinations = dict(examinations)

    @classmethod
    def load(cls, path: str | Path) -> "ExaminationIndicatorCatalog":
        with Path(path).open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise ValueError("Examination indicator catalog must be a JSON object")
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
            raise ValueError("Unsupported examination indicator catalog schema")
        rows = payload.get("examinations")
        if not isinstance(rows, dict):
            raise ValueError("examinations must be a JSON object")

        examinations: dict[str, tuple[ExaminationIndicator, ...]] = {}
        for raw_name, value in rows.items():
            name = str(raw_name).strip()
            if not name:
                raise ValueError("Examination name must not be blank")
            examinations[name] = _parse_examination(name, value)
        return cls(examinations)

    @property
    def examination_names(self) -> frozenset[str]:
        return frozenset(self._examinations)

    def indicators(self, examination: str) -> tuple[ExaminationIndicator, ...]:
        return self._examinations.get(str(examination), ())

    def applicable(
        self,
        examination: str,
        age_group: str,
        gender: str,
    ) -> tuple[ExaminationIndicator, ...]:
        return tuple(
            indicator
            for indicator in self.indicators(examination)
            if (not indicator.age_groups or age_group in indicator.age_groups)
            and (not indicator.genders or gender in indicator.genders)
        )


def _parse_examination(name: str, value: Any) -> tuple[ExaminationIndicator, ...]:
    if not isinstance(value, dict):
        raise ValueError(f"Catalog entry for {name} must be an object")
    rows = value.get("indicators")
    if not isinstance(rows, list):
        raise ValueError(f"Catalog entry for {name} must contain an indicators array")

    indicators: list[ExaminationIndicator] = []
    names: set[str] = set()
    for index, row in enumerate(rows):
        context = f"indicator {index} in {name}"
        if not isinstance(row, dict):
            raise ValueError(f"{context} must be an object")
        unexpected = sorted(set(row) - INDICATOR_FIELDS)
        if unexpected:
            raise ValueError(f"Unexpected field(s) in {context}: {', '.join(unexpected)}")
        indicator_name = str(row.get("name") or "").strip()
        if not indicator_name:
            raise ValueError(f"Missing name in {context}")
        if indicator_name in names:
            raise ValueError(f"Duplicate indicator name {indicator_name!r} in {name!r}")
        names.add(indicator_name)
        age_groups = _parse_values(row.get("age_groups"), AGE_GROUPS, "age_groups", context)
        genders = _parse_values(row.get("genders"), GENDERS, "genders", context)
        abnormal_case_only = row.get("abnormal_case_only")
        if not isinstance(abnormal_case_only, bool):
            raise ValueError(f"abnormal_case_only in {context} must be boolean")
        indicators.append(
            ExaminationIndicator(
                name=indicator_name,
                age_groups=age_groups,
                genders=genders,
                abnormal_case_only=abnormal_case_only,
            )
        )
    return tuple(indicators)


def _parse_values(value: Any, allowed: frozenset[str], field: str, context: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} in {context} must be an array")
    values = tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))
    invalid = sorted(set(values) - allowed)
    if invalid:
        raise ValueError(f"Invalid {field} in {context}: {', '.join(invalid)}")
    return values
