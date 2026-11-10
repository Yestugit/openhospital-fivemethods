from __future__ import annotations

import unittest
from pathlib import Path

from hospital_service.service.scripts.audit_examination_results import audit_examination_results


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class ExaminationResultAuditTest(unittest.TestCase):
    def test_full_data_has_no_examination_result_violations(self) -> None:
        report = audit_examination_results(REPOSITORY_ROOT)

        self.assertEqual(
            report["violations"],
            {name: 0 for name in report["violations"]},
            report.get("samples"),
        )


if __name__ == "__main__":
    unittest.main()
