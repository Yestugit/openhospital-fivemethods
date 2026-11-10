from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hospital_service.service.exam.catalog import ExaminationIndicatorCatalog


class ExaminationIndicatorCatalogTest(unittest.TestCase):
    def test_filters_explicit_age_and_gender_restrictions(self) -> None:
        catalog = self._load({
            "孕妇血清筛查": {
                "indicators": [
                    {
                        "name": "甲胎蛋白",
                        "age_groups": ["adult"],
                        "genders": ["female"],
                        "abnormal_case_only": False,
                    },
                    {
                        "name": "通用指标",
                        "age_groups": [],
                        "genders": [],
                        "abnormal_case_only": True,
                    },
                ]
            }
        })

        self.assertEqual(
            [item.name for item in catalog.applicable("孕妇血清筛查", "adult", "female")],
            ["甲胎蛋白", "通用指标"],
        )
        self.assertEqual(
            [item.name for item in catalog.applicable("孕妇血清筛查", "adult", "male")],
            ["通用指标"],
        )

    def test_rejects_duplicate_indicator_name(self) -> None:
        row = {
            "name": "甲胎蛋白",
            "age_groups": [],
            "genders": [],
            "abnormal_case_only": False,
        }
        with self.assertRaisesRegex(ValueError, "Duplicate indicator name"):
            self._load({"检查": {"indicators": [row, row]}})

    def test_rejects_source_and_identity_fields(self) -> None:
        row = {
            "name": "甲胎蛋白",
            "age_groups": [],
            "genders": [],
            "abnormal_case_only": False,
            "sources": ["normal_template"],
        }
        with self.assertRaisesRegex(ValueError, "Unexpected field"):
            self._load({"检查": {"indicators": [row]}})

    def _load(self, examinations: dict[str, object]) -> ExaminationIndicatorCatalog:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "examination_indicators.json"
        path.write_text(
            json.dumps({
                "metadata": {"schema_version": 1},
                "examinations": examinations,
            }, ensure_ascii=False),
            encoding="utf-8",
        )
        return ExaminationIndicatorCatalog.load(path)


if __name__ == "__main__":
    unittest.main()
