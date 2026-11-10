from __future__ import annotations

import unittest

from hospital_service.service.scripts.build_examination_indicators import (
    normalize_indicator_name,
)


class ExaminationIndicatorMigrationTest(unittest.TestCase):
    def test_normalizes_clear_afp_variants(self) -> None:
        for raw in ("甲胎蛋白（AFP）", "甲胎蛋白(AFP)", "AFP(甲胎蛋白)", "甲胎蛋白", "AFP"):
            with self.subTest(raw=raw):
                self.assertEqual(normalize_indicator_name(raw), "甲胎蛋白")

    def test_strips_clear_latin_abbreviation_suffix(self) -> None:
        self.assertEqual(normalize_indicator_name("C反应蛋白（CRP）"), "C反应蛋白")
        self.assertEqual(normalize_indicator_name("糖化血红蛋白(HbA1c)"), "糖化血红蛋白")

    def test_applies_reviewed_wording_correction(self) -> None:
        self.assertEqual(normalize_indicator_name("左心室壁厚"), "左心室壁厚度")
        self.assertEqual(normalize_indicator_name("沉深线"), "沉淀线")

    def test_does_not_merge_unreviewed_names(self) -> None:
        self.assertEqual(normalize_indicator_name("左心室壁厚度测量"), "左心室壁厚度测量")


if __name__ == "__main__":
    unittest.main()
