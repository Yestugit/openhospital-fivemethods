from __future__ import annotations

import asyncio
import unittest

from hospital_service.service.server import ContestServiceApp


class FakeExamService:
    def __init__(self) -> None:
        self.calls = 0

    async def get_results(self, payload):
        self.calls += 1
        return {
            "results": {
                "血常规": {
                    "result": {"血红蛋白": "145 g/L"},
                    "status": "normal",
                    "abnormal_indicators": [],
                }
            }
        }


class ExaminationServerTest(unittest.TestCase):
    def test_exam_route_awaits_async_service(self) -> None:
        exam_service = FakeExamService()
        services = type("Services", (), {"exam_service": exam_service})()
        app = ContestServiceApp.__new__(ContestServiceApp)
        app._post_context = lambda payload, headers: (object(), "team-a", services)  # type: ignore[method-assign]

        response = asyncio.run(app.handle_post_async(
            "/exam/results",
            {"team_id": "team-a", "patient_id": "Patient_001", "items": ["血常规"]},
            {},
        ))

        self.assertEqual(response["results"]["血常规"]["status"], "normal")
        self.assertEqual(exam_service.calls, 1)


if __name__ == "__main__":
    unittest.main()
