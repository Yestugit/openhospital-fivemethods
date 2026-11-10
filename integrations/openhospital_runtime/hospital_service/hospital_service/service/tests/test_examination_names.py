from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hospital_service.service.exam.names import ExaminationNames


class ExaminationNamesTest(unittest.TestCase):
    def test_resolves_historical_alias_and_deduplicates_requests(self) -> None:
        names = self._load([
            {"name": "肿瘤标志物谱", "aliases": ["肿瘤标志物检测"]},
        ])

        self.assertEqual(names.canonical("肿瘤标志物检测"), "肿瘤标志物谱")
        self.assertEqual(
            names.canonicalize_many(["肿瘤标志物检测", "肿瘤标志物谱"]),
            ["肿瘤标志物谱"],
        )

    def test_rejects_duplicate_canonical_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate canonical examination name"):
            self._load([{"name": "血常规"}, {"name": "血常规"}])

    def test_rejects_alias_pointing_to_two_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "Ambiguous examination alias"):
            self._load([
                {"name": "检查A", "aliases": ["旧检查"]},
                {"name": "检查B", "aliases": ["旧检查"]},
            ])

    def _load(self, rows: list[dict[str, object]]) -> ExaminationNames:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "examination_aliases.json"
        path.write_text(
            json.dumps({"examinations": rows}, ensure_ascii=False),
            encoding="utf-8",
        )
        return ExaminationNames.load(path)


if __name__ == "__main__":
    unittest.main()
