from __future__ import annotations

import unittest

from hospital_service.service.exam.result_store import (
    ExaminationResultRecord,
    InMemoryExaminationResultStore,
)


def _record(value: str, source: str = "generated") -> ExaminationResultRecord:
    return ExaminationResultRecord(
        dataset_key="train",
        patient_id="Patient_001",
        examination_name="血常规",
        result={"血红蛋白": value},
        indicator_statuses={"血红蛋白": "abnormal"},
        status="abnormal",
        abnormal_indicators=("血红蛋白",),
        source=source,  # type: ignore[arg-type]
        model_name="qwen3.5-flash" if source == "generated" else None,
    )


class ExaminationResultStoreTest(unittest.TestCase):
    def test_generated_result_requires_model_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "must include model_name"):
            ExaminationResultRecord(
                dataset_key="train",
                patient_id="Patient_001",
                examination_name="血常规",
                result={"血红蛋白": "88 g/L"},
                indicator_statuses={"血红蛋白": "abnormal"},
                status="abnormal",
                abnormal_indicators=("血红蛋白",),
                source="generated",
            )

    def test_put_keeps_first_concurrent_winner(self) -> None:
        store = InMemoryExaminationResultStore()
        first = store.put_if_absent(_record("88 g/L"))
        second = store.put_if_absent(_record("90 g/L"))

        self.assertEqual(second, first)

    def test_existing_import_does_not_replace_generated(self) -> None:
        store = InMemoryExaminationResultStore()
        generated = store.put_if_absent(_record("88 g/L"))
        imported = store.put_if_absent(_record("87 g/L", source="existing"))

        self.assertEqual(imported, generated)

    def test_returns_defensive_json_copy(self) -> None:
        store = InMemoryExaminationResultStore()
        store.put_if_absent(_record("88 g/L"))
        found = store.get("train", "Patient_001", "血常规")
        assert found is not None
        found.result["血红蛋白"] = "mutated"

        self.assertEqual(
            store.get("train", "Patient_001", "血常规").result["血红蛋白"],  # type: ignore[union-attr]
            "88 g/L",
        )


if __name__ == "__main__":
    unittest.main()
