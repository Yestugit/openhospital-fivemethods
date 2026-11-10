from __future__ import annotations

import asyncio
import unittest
from typing import Any

from hospital_service.service.exam.assessment import assess_examination_result
from hospital_service.service.exam.catalog import ExaminationIndicator, ExaminationIndicatorCatalog
from hospital_service.service.exam.generator import GeneratedExaminationResult
from hospital_service.service.exam.result_store import InMemoryExaminationResultStore
from hospital_service.service.exam.service import ExamService


class FakeRepository:
    dataset_key = "train"

    def __init__(self) -> None:
        self.patient_results = {
            "Patient_001": {
                "血常规": {
                    "血红蛋白": "88 g/L",
                    "白细胞": "12 x10^9/L",
                },
            },
        }

    def get_profile(self, patient_id: str) -> dict[str, Any]:
        if patient_id != "Patient_001":
            return {}
        return {"id": patient_id, "demographics": {"age": 40, "gender": "男"}}

    def get_patient_exam_result(self, patient_id: str, item_name: str) -> Any | None:
        return self.patient_results.get(patient_id, {}).get(item_name)

    def get_patient_exam_map(self, patient_id: str) -> dict[str, Any]:
        return dict(self.patient_results.get(patient_id, {}))

    def get_patient_exam_assessment(self, patient_id: str, item_name: str, result: Any):
        return assess_examination_result(result, self.get_normal_exam_result(patient_id, item_name))

    def get_normal_exam_result(self, patient_id: str, item_name: str) -> Any | None:
        normal = {
            "血常规": {"血红蛋白": "130-175 g/L", "白细胞": "4-10 x10^9/L"},
            "肝功能": {"丙氨酸氨基转移酶": "9-50 U/L"},
            "肾功能": {"血清肌酐": "57-97 umol/L"},
            "阴囊体格检查": {"阴毛": "变灰并变得稀疏。"},
        }
        return normal.get(item_name)

    def canonicalize_examination_items(self, items: list[str]) -> list[str]:
        aliases = {"旧肝功能": "肝功能"}
        result: list[str] = []
        for item in items:
            canonical = aliases.get(item, item)
            if canonical not in result:
                result.append(canonical)
        return result

    def get_patient_age_group(self, patient_id: str) -> str:
        return "adult"

    def get_patient_gender(self, patient_id: str) -> str:
        return "male"


class FakeGenerator:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def generate(self, **kwargs: Any) -> GeneratedExaminationResult:
        self.requests.append(kwargs)
        names = [indicator.name for indicator in kwargs["indicators"]]
        statuses = {name: "normal" for name in names}
        return GeneratedExaminationResult(
            result={name: f"{name}正常" for name in names},
            indicator_statuses=statuses,  # type: ignore[arg-type]
            status="normal",
            abnormal_indicators=(),
            model_name="fake-model",
            usage={},
        )


class MislabelingGenerator:
    async def generate(self, **kwargs: Any) -> GeneratedExaminationResult:
        return GeneratedExaminationResult(
            result={"丙氨酸氨基转移酶": "120 U/L"},
            indicator_statuses={"丙氨酸氨基转移酶": "normal"},
            status="normal",
            abnormal_indicators=(),
            model_name="fake-model",
            usage={},
        )


class ExamServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = FakeRepository()
        self.store = InMemoryExaminationResultStore()
        self.generator = FakeGenerator()
        self.catalog = ExaminationIndicatorCatalog({
            "肝功能": (ExaminationIndicator("丙氨酸氨基转移酶", (), (), False),),
            "肾功能": (ExaminationIndicator("血清肌酐", (), (), False),),
            "女性检查": (ExaminationIndicator("女性指标", (), ("female",), False),),
        })
        self.service = ExamService(
            self.repository,  # type: ignore[arg-type]
            catalog=self.catalog,
            result_store=self.store,
            generator=self.generator,  # type: ignore[arg-type]
        )

    def test_static_result_is_backfilled_and_reused(self) -> None:
        first = self._get(["血常规"])
        second = self._get(["血常规"])

        self.assertEqual(first, second)
        self.assertEqual(
            first["results"]["血常规"]["result"]["血红蛋白"],
            "88 g/L［参考值：130-175 g/L］",
        )
        self.assertEqual(self.generator.requests, [])
        stored = self.store.get("train", "Patient_001", "血常规")
        assert stored is not None
        self.assertEqual(stored.source, "existing")
        self.assertIsNone(stored.model_name)
        self.assertEqual(stored.status, "abnormal")
        self.assertEqual(stored.result["白细胞"], "12 x10^9/L［参考值：4-10 x10^9/L］")

    def test_missing_result_is_generated_once_and_reused(self) -> None:
        first = self._get(["旧肝功能"])
        second = self._get(["肝功能"])

        self.assertEqual(first, second)
        self.assertEqual(len(self.generator.requests), 1)
        self.assertEqual(list(first["results"]), ["肝功能"])
        self.assertEqual(
            first["results"]["肝功能"]["result"]["丙氨酸氨基转移酶"],
            "丙氨酸氨基转移酶正常［参考值：9-50 U/L］",
        )
        stored = self.store.get("train", "Patient_001", "肝功能")
        assert stored is not None
        self.assertEqual(stored.source, "generated")
        self.assertEqual(stored.model_name, "fake-model")

    def test_generation_context_includes_only_original_examination_results(self) -> None:
        self._get(["肝功能"])
        self._get(["肾功能"])

        known = self.generator.requests[-1]["known_results"]
        self.assertIn("血常规", known)
        self.assertNotIn("肝功能", known)

    def test_does_not_duplicate_existing_reference_text(self) -> None:
        self.repository.patient_results["Patient_001"]["血常规"]["血红蛋白"] = (
            "88 g/L［参考：130-175 g/L］"
        )

        response = self._get(["血常规"])

        value = response["results"]["血常规"]["result"]["血红蛋白"]
        self.assertEqual(value, "88 g/L［参考：130-175 g/L］")
        self.assertEqual(value.count("参考"), 1)

    def test_qualitative_normal_template_is_normal_without_reference_suffix(self) -> None:
        self.repository.patient_results["Patient_001"]["阴囊体格检查"] = {
            "阴毛": "变灰并变得稀疏。",
        }

        response = self._get(["阴囊体格检查"])

        examination = response["results"]["阴囊体格检查"]
        self.assertEqual(examination["status"], "normal")
        self.assertEqual(examination["result"]["阴毛"], "变灰并变得稀疏。")

    def test_generation_context_includes_age_and_gender_normal_reference(self) -> None:
        self._get(["肝功能"])

        self.assertEqual(
            self.generator.requests[-1]["normal_reference"],
            {"丙氨酸氨基转移酶": "9-50 U/L"},
        )

    def test_stores_model_generated_status_without_rule_reassessment(self) -> None:
        self.service.generator = MislabelingGenerator()  # type: ignore[assignment]

        response = self._get(["肝功能"])

        self.assertEqual(response["results"]["肝功能"]["status"], "normal")
        self.assertEqual(
            response["results"]["肝功能"]["abnormal_indicators"],
            [],
        )

    def test_canonicalizes_and_deduplicates_requested_examinations(self) -> None:
        response = self._get(["旧肝功能", "肝功能"])

        self.assertEqual(list(response["results"]), ["肝功能"])
        self.assertEqual(len(self.generator.requests), 1)

    def test_returns_invalid_for_unknown_or_inapplicable_examination(self) -> None:
        response = self._get(["不存在的检查", "女性检查"])

        for examination in ("不存在的检查", "女性检查"):
            self.assertEqual(response["results"][examination], {
                "result": "无效检查",
                "status": "invalid",
                "abnormal_indicators": [],
            })
        self.assertEqual(self.generator.requests, [])

    def test_rejects_unknown_patient(self) -> None:
        with self.assertRaisesRegex(KeyError, "Unknown patient_id"):
            asyncio.run(self.service.get_results({
                "patient_id": "missing",
                "items": ["血常规"],
            }))

    def _get(self, items: list[str]) -> dict[str, Any]:
        return asyncio.run(self.service.get_results({
            "patient_id": "Patient_001",
            "items": items,
        }))


if __name__ == "__main__":
    unittest.main()
