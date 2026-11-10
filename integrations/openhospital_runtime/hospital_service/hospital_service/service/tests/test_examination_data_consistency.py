from __future__ import annotations

import json
import unittest
from pathlib import Path

from hospital_service.service.exam.catalog import ExaminationIndicatorCatalog
from hospital_service.service.repository import ServiceDataRepository
from hospital_service.service.scripts.build_examination_indicators import (
    ExaminationIndicatorBuilder,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPOSITORY_ROOT / "hospital_service/service/data"


class ExaminationDataConsistencyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository = ServiceDataRepository(base_dir=DATA_ROOT, dataset_key="primary_test")
        cls.catalog = ExaminationIndicatorCatalog.load(DATA_ROOT / "examination_indicators.json")

    def test_catalog_build_is_idempotent(self) -> None:
        report = ExaminationIndicatorBuilder(REPOSITORY_ROOT).build(write=False)

        self.assertEqual(report["changes_required"], 0)
        self.assertGreater(report["examination_count"], 800)
        self.assertGreater(report["indicator_count"], 7000)

    def test_catalog_contains_no_source_or_identity_fields(self) -> None:
        payload = json.loads(
            (DATA_ROOT / "examination_indicators.json").read_text(encoding="utf-8")
        )
        serialized = json.dumps(payload, ensure_ascii=False)

        for forbidden in ("indicator_id", "scope", "aliases", "sources"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(f'"{forbidden}"', serialized)

    def test_afp_uses_bare_canonical_display_name(self) -> None:
        names = {
            item.name
            for item in self.catalog.applicable("甲胎蛋白（AFP）检测", "adult", "female")
        }

        self.assertIn("甲胎蛋白", names)
        self.assertNotIn("甲胎蛋白（AFP）", names)

    def test_old_examination_alias_resolves_to_canonical_name(self) -> None:
        self.assertEqual(
            self.repository.canonicalize_examination_items(["肿瘤标志物检测", "肿瘤标志物谱"]),
            ["肿瘤标志物谱"],
        )

    def test_repository_has_no_indicator_identity_or_overlay_indexes(self) -> None:
        for attribute in (
            "terminology",
            "overlay_permissions",
            "patient_indicator_occurrences",
            "examination_indicator_ids",
        ):
            with self.subTest(attribute=attribute):
                self.assertFalse(hasattr(self.repository, attribute))

    def test_direct_all_normal_sputum_culture_is_normal(self) -> None:
        repository = ServiceDataRepository(base_dir=DATA_ROOT, dataset_key="train")
        patient_id = "Patient_Comorbid-00023"
        examination = "痰培养"
        result = repository.get_patient_exam_result(patient_id, examination)
        assessment = repository.get_patient_exam_assessment(patient_id, examination, result)

        self.assertEqual(assessment.status, "normal")
        self.assertEqual(assessment.abnormal_indicators, ())


if __name__ == "__main__":
    unittest.main()
