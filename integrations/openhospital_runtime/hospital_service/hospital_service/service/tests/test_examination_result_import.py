from __future__ import annotations

import unittest
from typing import Any

from hospital_service.service.exam.assessment import assess_examination_result
from hospital_service.service.exam.result_store import (
    ExaminationResultRecord,
    InMemoryExaminationResultStore,
)
from hospital_service.service.scripts.import_examination_results import (
    import_repository_results,
)


class FakeRepository:
    dataset_key = "train"
    patient_exam_index = {
        "Patient_001": {
            "血常规": {"血红蛋白": "88 g/L"},
            "肝功能": {"丙氨酸氨基转移酶": "86 U/L"},
        }
    }

    def get_patient_exam_assessment(self, patient_id: str, examination: str, result: Any):
        normal = {
            "血常规": {"血红蛋白": "130-175 g/L"},
            "肝功能": {"丙氨酸氨基转移酶": "7-56 U/L"},
        }
        return assess_examination_result(result, normal[examination])


class ExaminationResultImportTest(unittest.TestCase):
    def test_imports_static_results_without_replacing_generated(self) -> None:
        store = InMemoryExaminationResultStore()
        store.put_if_absent(ExaminationResultRecord(
            dataset_key="train",
            patient_id="Patient_001",
            examination_name="肝功能",
            result={"丙氨酸氨基转移酶": "模型结果"},
            indicator_statuses={"丙氨酸氨基转移酶": "normal"},
            status="normal",
            abnormal_indicators=(),
            source="generated",
            model_name="model",
        ))

        report = import_repository_results(FakeRepository(), store)  # type: ignore[arg-type]

        self.assertEqual(report, {"inserted": 1, "preserved": 1})
        blood = store.get("train", "Patient_001", "血常规")
        liver = store.get("train", "Patient_001", "肝功能")
        assert blood is not None and liver is not None
        self.assertEqual(blood.source, "existing")
        self.assertEqual(blood.indicator_statuses, {"血红蛋白": "abnormal"})
        self.assertEqual(liver.source, "generated")

    def test_second_import_is_idempotent(self) -> None:
        store = InMemoryExaminationResultStore()
        repository = FakeRepository()

        first = import_repository_results(repository, store)  # type: ignore[arg-type]
        second = import_repository_results(repository, store)  # type: ignore[arg-type]

        self.assertEqual(first, {"inserted": 2, "preserved": 0})
        self.assertEqual(second, {"inserted": 0, "preserved": 2})


if __name__ == "__main__":
    unittest.main()
