"""Examination lookup and persistent lazy generation."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

try:
    from ..repository import ServiceDataRepository, normalize_items
except ImportError:
    from repository import ServiceDataRepository, normalize_items

from .catalog import ExaminationIndicatorCatalog
from .generator import ExaminationResultGenerator
from .result_store import (
    ExaminationResultRecord,
    ExaminationResultStore,
)

INVALID_EXAM_RESULT = "无效检查"
REFERENCE_MARKER = re.compile(r"(?:参考(?:范围|区间|值)|正常(?:范围|值)|参考[：:])")
NUMERIC_REFERENCE = re.compile(
    r"(?:[<>≤≥]\s*[-+]?\d|[-+]?\d+(?:\.\d+)?\s*(?:-|~|～|—|–|至)\s*[-+]?\d)"
)


class ExamService:
    def __init__(
        self,
        repository: ServiceDataRepository,
        *,
        catalog: ExaminationIndicatorCatalog,
        result_store: ExaminationResultStore,
        generator: ExaminationResultGenerator,
    ):
        self.repository = repository
        self.catalog = catalog
        self.result_store = result_store
        self.generator = generator

    async def get_results(self, payload: dict[str, Any]) -> dict[str, Any]:
        patient_id = _required_text(payload, "patient_id")
        raw_items = payload.get("items")
        if not isinstance(raw_items, list):
            raise ValueError("items must be a JSON array.")
        requested = self.repository.canonicalize_examination_items(normalize_items(raw_items))
        if not self.repository.get_profile(patient_id):
            raise KeyError(f"Unknown patient_id: {patient_id}")

        results: dict[str, dict[str, Any]] = {}
        for examination in requested:
            results[examination] = await self._get_one(patient_id, examination)
        return {"results": results}

    async def _get_one(self, patient_id: str, examination: str) -> dict[str, Any]:
        normal_reference = self.repository.get_normal_exam_result(patient_id, examination)
        stored = await asyncio.to_thread(
            self.result_store.get,
            self.repository.dataset_key,
            patient_id,
            examination,
        )
        if stored is not None:
            return _public_result(stored, normal_reference)

        static_result = self.repository.get_patient_exam_result(patient_id, examination)
        if static_result is not None:
            assessment = self.repository.get_patient_exam_assessment(
                patient_id,
                examination,
                static_result,
            )
            static_result = _with_normal_references(static_result, normal_reference)
            record = ExaminationResultRecord(
                dataset_key=self.repository.dataset_key,
                patient_id=patient_id,
                examination_name=examination,
                result=static_result,
                indicator_statuses=assessment.indicator_statuses,
                status=assessment.status,
                abnormal_indicators=assessment.abnormal_indicators,
                source="existing",
            )
            stored = await asyncio.to_thread(self.result_store.put_if_absent, record)
            return _public_result(stored, normal_reference)

        indicators = self.catalog.applicable(
            examination,
            self.repository.get_patient_age_group(patient_id),
            self.repository.get_patient_gender(patient_id),
        )
        if not indicators:
            return _invalid_result()

        generated = await self.generator.generate(
            examination=examination,
            indicators=indicators,
            profile=self.repository.get_profile(patient_id),
            known_results=self.repository.get_patient_exam_map(patient_id),
            normal_reference=normal_reference,
        )
        generated_result = _with_normal_references(generated.result, normal_reference)
        record = ExaminationResultRecord(
            dataset_key=self.repository.dataset_key,
            patient_id=patient_id,
            examination_name=examination,
            result=generated_result,
            indicator_statuses=generated.indicator_statuses,
            status=generated.status,
            abnormal_indicators=generated.abnormal_indicators,
            source="generated",
            model_name=generated.model_name,
        )
        stored = await asyncio.to_thread(self.result_store.put_if_absent, record)
        return _public_result(stored, normal_reference)


def _public_result(
    record: ExaminationResultRecord,
    normal_reference: Any | None,
) -> dict[str, Any]:
    return {
        "result": _with_normal_references(record.result, normal_reference),
        "status": record.status,
        "abnormal_indicators": list(record.abnormal_indicators),
    }


def _with_normal_references(result: Any, normal_reference: Any | None) -> Any:
    if not isinstance(result, dict) or not isinstance(normal_reference, dict):
        return result

    formatted = dict(result)
    for name, reference in normal_reference.items():
        if name not in formatted or _is_blank(reference) or not _is_numeric_reference(reference):
            continue
        value_text = _display_value(formatted[name])
        if REFERENCE_MARKER.search(value_text):
            continue
        formatted[name] = f"{value_text}［参考值：{_display_value(reference)}］"
    return formatted


def _display_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _is_numeric_reference(value: Any) -> bool:
    return isinstance(value, str) and NUMERIC_REFERENCE.search(value) is not None


def _invalid_result() -> dict[str, Any]:
    return {
        "result": INVALID_EXAM_RESULT,
        "status": "invalid",
        "abnormal_indicators": [],
    }


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"{key} is required.")
    return text
